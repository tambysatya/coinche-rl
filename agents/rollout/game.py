import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as rnd
import jax.tree_util as jtu
import flax.nnx as nnx


from agents.rollout.bid import *
from agents.rollout.trick import *
from coinche.Hand import *


def mk_rollout(bid_policy, game_policy, pool_size):
    bid_rollout = mk_bid_rollout (bid_policy, pool_size)
    game_rollout = mk_game_rollout(game_policy, pool_size)


    def rollout(all_bid_params, all_game_params, 
                permutation,
                hidden_states,
                initial_player,
                seed):
        """ Simulates an entire game:
                output:
                    - traj_tricks : completes tricks where all 4 people played : [8 x batch_size]
                    - bidding_steps : bidding observations of all players [10 x 4 x batch_size]
                    - trick_steps : observations of all players during the trick plays [8 x 4 x batch_size]
        """
        batch_size = initial_player.shape[0]

        deal_seed, bid_seed, trick_seed = rnd.split(seed, 3)
        hands = deal(rnd.split(deal_seed, batch_size))

        rollout  = bid_rollout(all_bid_params,
                               permutation,
                               hands,
                               hidden_states,
                               initial_player,
                               bid_seed)

        hidden_states, bidding_count, bid, bidding_steps = rollout #everything is [B, ....] except bididng_steps which is [40, B, ....]
    
        final_history, traj_tricks,  traj_steps = game_rollout(all_game_params,
                                               permutation,
                                               hidden_states, 
                                               bid, hands,
                                               trick_seed)

        bidding_steps = jtu.tree_map(lambda l: l.reshape(10,4,*l.shape[1:]), bidding_steps)
        return final_history, traj_tricks, bidding_steps, traj_steps
    return jax.jit(rollout)

                         
