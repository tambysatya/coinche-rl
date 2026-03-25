from flax import struct
import jax
import jax.numpy as jnp
import jax.random as rnd
from jaxtyping import Array, Bool, Int



TensorBid = Bool [Array, "B 62"] # 60 + Coinche + Overcoinche

@struct.dataclass
class Bid:
    bid : Bool [Array, "B 6 10"]
    coincheP : Bool [Array, "B"]
    overcoincheP : Bool [Array, "B"]

@jax.jit
def bid_to_tensor(bid : Bid) -> TensorBid:
    tensor = bid.bid.reshape(-1, 60)
    return jnp.concatenate([tensor, bid.coincheP, bid.overcoincheP], axis=1)

@jax.jit
def bid_from_tensor(tensor: TensorBid) -> Bid:
    bid, coincheP, overcoincheP = tensor[:, :60], tensor[:, 60], tensor[:, 61]
    return Bid(bid.reshape(-1, 6, 10), coincheP, overcoincheP)


@jax.jit
def possible_bid_mask (tensor : TensorBid) -> TensorBid:
    """ Generates the possible bidding actions (i.e. coincheP = 1 => you are allowed to coinche)"""
    batch_size = tensor.shape[0]

    bid = bid_from_tensor(tensor)
    coinche_tensor = jnp.ones([batch_size], dtype=bool).reshape(-1,1)
    overcoinche_tensor = jnp.ones([batch_size], dtype=bool).reshape(-1,1)

    all_ranks = jnp.arange(10).reshape(1,1,10).repeat(6, axis=1).repeat(batch_size, axis=0)
    current_rank = bid.bid.sum(axis=1).argmax(axis=1)
    above_bid = jnp.where(all_ranks > current_rank, True, False)

    

    above_bid_mask = Bid((1-bid.coincheP)*above_bid,
                     (1-bid.coincheP)*coinche_tensor, 
                     overcoinche_tensor)
    above_bid_mask = bid_to_tensor(above_bid_mask)
    return above_bid_mask
    

