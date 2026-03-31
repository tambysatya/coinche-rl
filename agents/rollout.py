from flax import struct
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as rnd
import jax.tree_util as jtu
import flax.nnx as nnx

from coinche.Trick import *
from coinche.LegalMoves import *

@struct.dataclass
class Step:
    # Inputs of the policy network
    obs : jax.Array # observation of the state 
    hidden_state : jax.Array # embedding of the past experiences (to augment the state with an embedding, produced by the previous calls of the policy network, and carried through the trajectory).
    #Action 
    logprobs : jax.Array # log-probability inferred by the network


def mk_step(policy_model):
    graphdef, _ = nnx.split(policy_model)

    def step (params, hidden_state,
              trump : Suit, trick : Trick,
              key):
        obs = trick_obs(trick)
        policy = nnx.merge(graphdef, params)

        legal_moves = possible_moves(trump, trick)
        logits, next_hidden_state = policy(trump, obs, hidden_state)
        probas = jnp.where(legal_moves.reshape([-1,32]),
                           logits,
                           -jnp.inf)
        action = rnd.categorical(key, probas)
        card = card_from_index(action)

        record = Step(obs, next_hidden_state, jnp.log(probas))

        return play(trump, trick, card), record
    
    return jax.jit(step)

        
def mk_trick_rollout (policy_model):
    """ Plays a full trick (the 4 players chose a card) """
    step = mk_step(policy_model)

    def trick_rollout (params,
                       initial_step, # dummy record (for the first trick) or the output of past iterations
                       trump : Suit, initial_player : Player, hands : Hand,
                       seed) -> Trick:
        trick = new_trick(initial_player, hands)


        def scan_step (carry, step_seed):
            prev_trick, prev_records = carry
            new_trick, new_record = step(params, prev_records.hidden_state,
                                         trump, prev_trick, step_seed)
            return (new_trick, new_record) , new_record

        final, trajectory_records = jax.lax.scan(scan_step, (trick, initial_step), rnd.split(seed, 4) )
        final_trick, final_record = final

        return (final_trick, final_record), trajectory_records


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
        dummy_step = Step(jnp.zeros([batch_size, 97]), # dummy observation
                          initial_hidden_state,   # hidden state after the bidding phase
                          jnp.zeros([batch_size, 32])) # dummy logprob
        initial_trick = new_trick (initial_player, initial_hands)


        def scan_step (carry, trick_seed):
            prev_final_trick, prev_final_record = carry 
            (final_trick, final_record), trajectory_records = trick_rollout (
                                                                  params, prev_final_record, trump,
                                                                  prev_final_trick.best_player,
                                                                  prev_final_trick.hands,
                                                                  trick_seed)
            return (final_trick, final_record), (final_trick, trajectory_records)


        (final_trick, final_record), (traj_trick, traj_records) = jax.lax.scan(scan_step,  (initial_trick, dummy_step), rnd.split(seed, 8))
        return traj_trick, traj_records

    return jax.jit(rollout)

