from flax import struct
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as rnd
import jax.tree_util as jtu
import flax.nnx as nnx

from coinche.Trick import *
from coinche.Bid import *
from coinche.LegalMoves import *


Score = Int [Array, "B"]

@struct.dataclass
class Observation:
    """ Synthesis of the observations of a trick """
    bid : BidHistory
    trick : TrickObs # actual trick, with only the hand of the current player
    hidden_state : jax.Array # user-defined datastructure that is both used and produced by the policy/value network to augment the data

@struct.dataclass
class TrickStep:
    # Inputs of the policy network
    obs : Observation # observation of the state 
    #Action 
    agent : Int [Array, "B"] #agent index
    action : Int [Array, "B"] #chosen card
    action_mask : Bool [Array, "B 32"]
    logprobs : jax.Array #log-probability of the chosen action



def mk_step(policy_model):
    graphdef, _ = nnx.split(policy_model)

    def step (agent_index,  # current agent index : Int
              params, hidden_state,
              bid : BidHistory, trick : TrickHistory,
              key):
        """ The current player plays a card """

        obs = Observation(bid, trick_history_obs(trick), hidden_state)
        policy = nnx.merge(graphdef, params)

        legal_moves = possible_moves(trick)
        action_mask = legal_moves.reshape([-1,32])

        logits, next_hidden_state = policy(bid, obs)

        logits = jnp.where(action_mask,
                           logits,
                           -jnp.inf)
        action = rnd.categorical(key, logits)
        card = card_from_index(action)

        probas = jax.vmap(lambda p, a: p[a])(jax.nn.softmax(logits), action)

        batch_size = action.shape[0]
        record = TrickStep(obs,
                           jnp.tile (agent_index, (batch_size, 1)), #also store the index of the agent who plays (same agent for the entire batch)
                           action, action_mask, jnp.log(probas))

        return (next_hidden_state, play(trick, card)), record

    return jax.jit(step)


def mk_league_step (policy_model, pool_size):

    agent_step = mk_step(policy_model)

    def step (all_params, # [P, Params]
              permutation : Int [Array, "B"], # [B] : index of the agent that plays this turn
              hidden_state, #[B, ...] hidden state of the agent
              history : BidHistory, trick : TrickHistory, 
              key):
        
        batch_per_agent = group_dataset_by_agent(pool_size, permutation, (hidden_state, history, trick))
        hidden_state, history, trick = batch_per_agent # [(P, B, ...)]

        ret, record = jax.vmap(agent_step)(
                jnp.arange(pool_size), all_params,
                hidden_state, history, trick, rnd.split(key,pool_size))

        ret, record = ungroup_dataset_by_agent(permutation, (ret,record))

        return ret, record

    return jax.jit(step)
        

        
def mk_trick_rollout (policy_model, pool_size):
    """ Plays a full trick (the 4 players chose a card) """
    step = mk_league_step(policy_model, pool_size)

    def trick_rollout (all_params,
                       permutations: Int [Array, "B 2"], # permutation describing which agents plays Team0 and Team1
                       initial_hidden, #  [B, 4, ...] previous hidden state for each player
                       bid : BidHistory,
                       trick : TrickHistory,
                       seed) -> Trick:



        def scan_step (carry, step_seed):
            all_hidden, prev_trick = carry
            cur_team = prev_trick.current_player % 2
            player_permutation = jax.vmap(lambda a, p: a[p])(permutations, cur_team) # extracts the scores and index of the current player
            player_hidden = jax.vmap(lambda h, p: h[p])(all_hidden, prev_trick.current_player) #extracts the hidden state of the current player

            (new_hidden, new_trick), new_record = step(all_params, player_permutation,
                                                       player_hidden,
                                                       bid, prev_trick,
                                                       step_seed)
            new_hidden = jax.vmap(lambda a, h, p: a.at[p].set(h))(all_hidden, new_hidden, prev_trick.current_player) #updates the hidden state of the playing agent
            return (new_hidden, new_trick) , new_record

        final, trajectory_records = jax.lax.scan(scan_step, (initial_hidden, trick), rnd.split(seed, 4) )
        final_hidden, final_trick = final

        return (final_hidden, final_trick), trajectory_records


    return jax.jit(trick_rollout)
        

def mk_game_rollout (policy_model, pool_size):
    trick_rollout = mk_trick_rollout(policy_model, pool_size)

    def rollout (all_params, # Params of each agent of the pool: [P, ...]
                 permutations : Int [Array, "B 2"], # Pair of agent for each game
                 initial_hidden_state, # [B, 4, ...] hidden state of each player, for each game
                 bid : BidHistory, initial_hands : Hand,
                 seed):
        """ Simulates a complete trick phase (8 tricks):
                input : - parameters of the each policy network
                        - sorted permutation of indices describing which agent plays on each game (obtained by argsorting the indices of which agent play each team of each game)
                        - batch of 4 user-defined hidden_state, also passed to the policy network
                        - initial conditions of the game: trump suit, starting player, and cards distributions
                returns: - the complete tricks among the trajectory: it is the resulting trick where all 4 players played a card
                         - the observations records among the trajectory (4 per trick)
                Dimension:
                    - complete tricks : [8 x batch_size]
                    - observations records : [8 x 4 x batch_size]
        """
        
        trick = trick_history_initialize(bid, initial_hands)

        def scan_step (carry, trick_seed):
            prev_hidden, prev_trick = carry 
            (next_hidden, finished_trick), trajectory_records = trick_rollout (
                                                                  all_params, permutations,
                                                                  prev_hidden,
                                                                  bid, prev_trick,
                                                                  trick_seed)

            return (next_hidden, finished_trick), (finished_trick, trajectory_records)

        

        (final_trick, final_record), (traj_trick, traj_records) = jax.lax.scan(scan_step,  (initial_hidden_state, trick), rnd.split(seed, 8))
        return traj_trick, traj_records

    return jax.jit(rollout)


@jax.jit
def transition_rewards (trump : Suit, # [batch_size]
                        traj_trick : Trick, # [8, batch_size]
                        traj_records : TrickStep): # [8, 4, batch_size]
    """ Generates a batch of example (TrickStep, TransitionReward), both are of shape [8, 4, batch_size]"""

    batch_size = trump.shape[0]
    # adding the 10 de der (last trick amounts 10 additional points, except in ALL_TRUMP)
    has_10_der_p = ~(trump == SUIT_ALL_TRUMP)
    bonus = has_10_der_p*10 
    values = traj_trick.value.at[-1,:].add(bonus) # [8, batch_size]
    winners = traj_trick.best_player % 2 # [8, batch_size]

    
    player_indices = jnp.arange(4)[None,:,None]
    player_indices = jnp.tile(player_indices, (8, 1, batch_size)) #(8,4,batch_size) => player_indices[a][b] = [b,b,b,...,b] for each item of the batch
    winners = jnp.tile(winners[:,None,:], (1,4,1)) # (8, 4, batch_size)
    values = jnp.tile(values[:,None,:], (1,4,1)) # (8, 4, batch_size)

    transition_rewards = jnp.where(player_indices % 2 == winners, values, 0) # (8, 4, batch_size)

    # generates the dataset 
    #rewards = rewards.flatten() # [B*32, 1]
    #traj_records = jtu.tree_map(lambda l: l.reshape([batch_size*8*4,-1]),traj_records) #[B*32,...]



    return transition_rewards


@jax.jit
def cumulative_rewards (transition_rewards, # [8, 4, batch_size]
                        discount_factor): # float
    """ Generates the state/values function, ie the cumulative rewards (or cost-to-go) from each state:
        sum_i gamma^i r_i where gamma is the discount factor"""

    def cumulative_sum (i):
        # sets to 0 every index before i 
        coefs = jnp.arange(8) - i 
        coefs = jnp.maximum(coefs, 0)
        coefs = discount_factor ** coefs

        rew = transition_rewards * coefs[:,None,None]

        discounted_rew = jnp.cumsum(rew, axis=0)
        return discounted_rew[i]

    return jax.vmap(cumulative_sum)(jnp.arange(8))


def mk_collect_game_samples(policy_mdl, pool_size):
    rollout = mk_game_rollout(policy_mdl, pool_size)

    def collect_samples (
            discount_factor,
            all_params,
            agent_indices : Int [Array, "B 2"], # specifies which pair of agent plays each game
            initial_hidden_state,
            trump : Suit, initial_player : Player, initial_hands : Hand,
            seed):
        """ Samples rollouts, and generates a dataset [state, cumulative_reward] """

        batch_size = trump.shape[0]

        permutations = agent_indices.argsort(axis=1) # computes one for all the permutations in order to easily reorder the batch by consecutive players

        traj_tricks, traj_records = rollout(all_params, permutations,
                                            initial_hidden_state,
                                            trump, initial_player, initial_hands, seed)
        rewards = transition_rewards(trump, traj_tricks, traj_records)
        rewards = cumulative_rewards(rewards, discount_factor)

        return jtu.tree_map(lambda l: l.reshape([batch_size*32,-1]).squeeze(), traj_records), rewards.flatten()
    return jax.jit(collect_samples)


