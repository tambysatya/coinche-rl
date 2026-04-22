import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as rnd
import jax.tree_util as jtu
import flax.nnx as nnx

from functools import partial

from utils import *
from coinche.Card import *
from agents.rollout.game import *


def mk_compute_rewards(pool_size):
    def compute_transition_rewards (dicount_factor : Int,
                                    trick_hist : TrickHistory, # 8,B 
                                    bidding_steps : BidStep, # 10,4,B
                                    traj_steps : TrickStep): # 8,4,B
        last_entry = traj_hist.team_scores[-1]
        has_tenth_of_der = traj_tricks.trump[-1] != SUIT_ALL_TRUMP
        last_winner = traj_tricks.tric


        return last_entry, has_tenth_of_der




    return jax.jit(compute_transition_rewards)


