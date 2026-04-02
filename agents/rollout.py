from flax import struct
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as rnd
import jax.tree_util as jtu
import flax.nnx as nnx

from coinche.Trick import *
from coinche.LegalMoves import *


Score = Int [Array, "B"]

@struct.dataclass
class Observation:
    """ Synthesis of the observations of a trick """
    trick : jax.Array # actual trick, with only the hand of the current player
    current_score : Score  # current score of the team
    total_score : Score # sum of the cards played so far
    hidden_state : jax.Array # user-defined datastructure that is both used and produced by the policy/value network to augment the data

@struct.dataclass
class Step:
    # Inputs of the policy network
    obs : Observation # observation of the state 
    #Action 
    action : Int [Array, "B"] #chosen card
    action_mask : Bool [Array, "B 32"]
    logprobs : jax.Array #log-probability of the chosen action



def mk_step(policy_model):
    graphdef, _ = nnx.split(policy_model)

    def step (params, hidden_state,
              trump : Suit, trick : Trick,
              current_score,total_score, 
              key):
        """ The current player plays a card """
        obs = Observation(trick_obs(trick), current_score, total_score, hidden_state)
        policy = nnx.merge(graphdef, params)

        legal_moves = possible_moves(trump, trick)
        action_mask = legal_moves.reshape([-1,32])
        logits, next_hidden_state = policy(trump, obs)
        logits = jnp.where(action_mask,
                           logits,
                           -jnp.inf)
        action = rnd.categorical(key, logits)
        card = card_from_index(action)

        probas = jax.vmap(lambda p, a: p[a])(jax.nn.softmax(logits), action)

        record = Step(obs, action, action_mask, jnp.log(probas))

        return (next_hidden_state, play(trump, trick, card)), record
    
    return jax.jit(step)

        
def mk_trick_rollout (policy_model):
    """ Plays a full trick (the 4 players chose a card) """
    step = mk_step(policy_model)

    def trick_rollout (params,
                       initial_hidden, # previous hidden state
                       trump : Suit,
                       team_score, # [score_team_0, score_team_1]
                       total_score,
                       initial_player : Player, hands : Hand,
                       seed) -> Trick:
        trick = new_trick(initial_player, hands)
        


        def scan_step (carry, step_seed):
            prev_hidden, prev_trick = carry
            player_score = jax.vmap(lambda s, p: s[p])(team_score, prev_trick.current_player%2) # scores of the current player

            (new_hidden, new_trick), new_record = step(params, prev_hidden,
                                                       trump, prev_trick,
                                                       player_score, total_score,
                                                       step_seed)
            return (new_hidden, new_trick) , new_record

        final, trajectory_records = jax.lax.scan(scan_step, (initial_hidden, trick), rnd.split(seed, 4) )
        final_hidden, final_trick = final

        return (final_hidden, final_trick), trajectory_records


    return jax.jit(trick_rollout)
        

def mk_rollout (policy_model):
    trick_rollout = mk_trick_rollout(policy_model)

    def rollout (params,
                 initial_hidden_state,
                 trump : Suit, initial_player : Player, initial_hands : Hand,
                 seed):
        """ Simulates a complete trick phase (8 tricks):
                input : - parameters of the policy network
                        - user-defined hidden_state, also passed to the policy network
                        - initial conditions of the game: trump suit, starting player, and cards distributions
                returns: - the complete tricks among the trajectory: it is the resulting trick where all 4 players played a card
                         - the observations records among the trajectory (4 per trick)
                Dimension:
                    - complete tricks : [8 x batch_size]
                    - observations records : [8 x 4 x batch_size]
        """
        batch_size = trump.shape[0]
        initial_scores = jnp.zeros([batch_size,2])
        initial_total_score = jnp.zeros([batch_size])
        initial_trick = new_trick (initial_player, initial_hands)


        def scan_step (carry, trick_seed):
            prev_hidden, prev_trick, prev_scores, prev_total_score = carry 
            (next_hidden, finished_trick), trajectory_records = trick_rollout (
                                                                  params, prev_hidden,
                                                                  trump,
                                                                  prev_scores, #score of both teams
                                                                  prev_total_score,
                                                                  prev_trick.best_player,
                                                                  prev_trick.hands,
                                                                  trick_seed)
            #update the team scores according to which team won the trick
            new_score = jnp.where((finished_trick.best_player % 2 == 0)[:,None],
                                  prev_scores.at[:,0].add(finished_trick.value),
                                  prev_scores.at[:,1].add(finished_trick.value))
            total_score = prev_total_score + finished_trick.value


            return (next_hidden, finished_trick, new_score, total_score), (finished_trick, trajectory_records)

        

        (final_trick, final_record, _, _), (traj_trick, traj_records) = jax.lax.scan(scan_step,  (initial_hidden_state, initial_trick, initial_scores, initial_total_score), rnd.split(seed, 8))
        return traj_trick, traj_records

    return jax.jit(rollout)


@jax.jit
def transition_rewards (trump : Suit, # [batch_size]
                        traj_trick : Trick, # [8, batch_size]
                        traj_records : Step): # [8, 4, batch_size]
    """ Generates a batch of example (Step, TransitionReward), both are of shape [8, 4, batch_size]"""

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


def mk_collect_samples(policy_mdl):
    rollout = mk_rollout(policy_mdl)

    def collect_samples (
            discount_factor,
            params,
            initial_hidden_state,
            trump : Suit, initial_player : Player, initial_hands : Hand,
            seed):
        """ Samples rollouts, and generates a dataset [state, cumulative_reward] """

        batch_size = trump.shape[0]
        traj_tricks, traj_records = rollout(params, initial_hidden_state,
                                            trump, initial_player, initial_hands, seed)
        rewards = transition_rewards(trump, traj_tricks, traj_records)
        rewards = cumulative_rewards(rewards, discount_factor)

        return jtu.tree_map(lambda l: l.reshape([batch_size*32,-1]).squeeze(), traj_records), rewards.flatten()
    return jax.jit(collect_samples)


