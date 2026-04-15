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
        batch_size = initial_player.shape[0]

        deal_seed, bid_seed, trick_seed = rnd.split(seed, 3)
        hands = deal(rnd.split(deal_seed, batch_size))

        rollout  = bid_rollout(all_bid_params,
                               permutation,
                               hands,
                               hidden_states,
                               initial_player,
                               bid_seed)

        print ("Bidding phase")
        hidden_states, bidding_count, history, bidding_steps = rollout #everything is [B, ....] except bididng_steps which is [40, B, ....]
        trick_history = trick_history_initialize(history, hands)

        print ("Playing phase")
        trump = history_current_record(history).bid.rank.argmax(axis=-1)
        traj_tricks, traj_records = game_rollout(all_game_params,
                                                 permutation,
                                                 hidden_states,
                                                 trump, history, initial_player,hands,
                                                 trick_seed)

        return traj_tricks, bidding_steps, traj_records
    return jax.jit(rollout)

                         
