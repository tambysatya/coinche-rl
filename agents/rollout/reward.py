import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as rnd
import jax.tree_util as jtu
import flax.nnx as nnx

from functools import partial

from utils import *
from agents.rollout.game import *


def mk_compute_rewards(pool_size):
    def compute_rewards_of_team (team_permutation : Int [Array, "B"],
                                 dicount_factor : Int,
                                 traj_tricks : Trick, # 8,B 
                                 bidding_steps : BidStep, # 10,4,B
                                 traj_steps : TrickStep): # 8,4,B

        group = partial(group_dataset_by_agent, pool_size, team_permutation)
        traj_tricks = jtu.tree_map(jax.vmap(group), traj_tricks)
        bidding_steps = jtu.tree_map(jax.vmap(jax.vmap(group)), bidding_steps)
        traj_steps = jtu.tree_map(jax.vmap(jax.vmap(group)), traj_steps)

        return traj_tricks, bidding_steps, traj_steps

    return jax.jit(compute_rewards_of_team)


