import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as rnd
import jax.tree_util as jtu
import flax.nnx as nnx


from agents.rollout.bid import *
from agents.rollout.trick import *


def mk_rollout(bid_policy, game_policy, pool_size):
    bid_rollout = mk_bid_rollout (bid_policy, pool_size)
    game_rollout = mk_game_rollout(game_policy, pool_size)


    def rollout(bid_all_params, game_all_params, 
                permutation,
                hidden_states,
                initial_player,
                seed):

                         
