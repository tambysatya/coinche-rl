
from flax import struct
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as rnd
import jax.tree_util as jtu
import flax.nnx as nnx


from coinche.Bid import *


@struct.dataclass
class BidObs:
    """A partial observation of a bid"""
    bid : Bid
    author :  Int [Array, "B"] #author of the bid
    checked : Bool [Array, "B 4"] #one-hot encoding of which player has already checked



def mk_bid_rollout(bidding_model, pool_size):
    graphdef, _ = nnx.split(bidding_model)


    def bid_rollout (all_params,
                     permutation : Int[Array, "B 2"],
                     hidden_state : jax.Array,
                     current_player : Int [Array, "B"],
                     obs : BidObs):
        permutation, hidden_state = jax.vmap(lambda i,h,p : (p[i], h[i]))(current_player, hidden_state, permutation)
        current_player, obs = group_dataset_by_agent(all_params, permutation, (current_player, obs))
        output, hidden_state = jax.vmap


