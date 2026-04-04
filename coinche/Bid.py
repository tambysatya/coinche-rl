from flax import struct
import jax
import jax.numpy as jnp
import jax.random as rnd
import jax.tree_util as jtu
from jaxtyping import Array, Bool, Int



TensorBid = Bool [Array, "B 62"] # 60 + Coinche + Overcoinche

@struct.dataclass
class Bid:
    """ A bid made by a player """
    suit : Int [Array, "B"] # [0,5] total 6: all 4 suits + ALL_TRUMP + NO_TRUMP 
    rank : Int [Array, "B"] #0-8 total 9: from 80 to 160 (8 calls) + ALL_IN 
    check : Bool [Array, "B"]
    coincheP : Bool [Array, "B"]
    overcoincheP : Bool [Array, "B"]


@struct.dataclass
class BidMask:
    """ Masks on the possible bids """
    possible_rank : Bool [Array, "B 6"]
    possible_check : Bool [Array, "B"]
    possible_coinche : Bool [Array, "B"]
    possible_overcoinche : Bool [Array, "B"]




@jax.jit
def bid_to_tensor (bid : Bid):
    return jnp.concatenate([jax.nn.one_hot(bid.suit, 6),
                            jax.nn.one_hot(bid.rank, 9),
                            bid.check[:,None],
                            bid.coincheP[:,None],
                            bid.overcoincheP[:,None]], axis=1)

@jax.jit
def bids_above (bid : Bid) -> BidMask: 
    """ Given a batch of bid, gives a mask of all the allowed bids """
    
    batch_size = bid.suit.shape[0]

    emptyval = jnp.array([-1])
    true = jnp.ones([batch_size], dtype=bool)
    false = jnp.zeros([batch_size], dtype=bool)



    # if the player can raise, he can also check or coinche
    raised_mask = BidMask (jnp.tile(jnp.arange(0, 9),(batch_size,1)) > bid.rank,
                      true, true, false) #
    # if the game has been coinched, he can either overcoinche or check
    coinched_mask = BidMask (jnp.zeros([batch_size, 9], dtype=bool),true, false, true) #if coinched, you can only overcoinche

    def new_bid(raised, coinched):
        return jnp.where(
                    bid.coincheP,
                    coinched,
                    raised)

    return jtu.tree_map (new_bid, raised_bid, coinched_bid)

    


   

