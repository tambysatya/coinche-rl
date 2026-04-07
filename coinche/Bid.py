from flax import struct
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jax.random as rnd
import jax.tree_util as jtu
from jaxtyping import Array, Bool, Int, Float

from utils import *
from coinche.Hand import *

# Bidding system:
#   - 3 policy network: for bidding, coinching and overcoinching
# Representations:
#   - Total representation : [Suit, Rank, Who checked, Who coinched, Who overcoinched]
#   - for bidding and coinching : [Suit, Rank, Who checked]
#   - for overcoinching : [Suit, Rank, Who checked, Who coinched]
# Total Rollout:
#   - Someone enounce a bidding (can be a "check")
#   - Coinche Rollout (only taken into consideration if the bidding is not a "check"):
#               - Each opponent has the opportunity to coinche
#   - Overcoinche Rollout (only taken into consideration if an opponent has coinched):
#               - Each member of the bidding team has the opportunity to overcoinched



TensorBid = Bool [Array, "B 62"] # 60 + Coinche + Overcoinche

@struct.dataclass
class Bid:
    """ A bid made by a player
        The "PASS" bid is equal to "False" everywhere
    """
    suit : Bool [Array, "B 6"] # one-hot [0,5] total 6: all 4 suits + ALL_TRUMP + NO_TRUMP 
    rank : Bool [Array, "B 9"] # one-hot 0-8 total 9: from 80 to 160 (8 calls) + ALL_IN 
    author : Int [Array, "B"] # index of the player who called this bid
    passed : Bool [Array, "B 4"] # who passed on the bid

@struct.dataclass
class BidHistory:
    """ Represents all the previous bid, and a pointer to the current bid:
        Sice ranks can go from 80 to 160 (total=8) + allin (total=9), as most 9 possible bid can be called in a single game.
    """
    entries : Bid # B 9 Bid
    index : Int [Array, "B"] # index of the current bid

def bid_to_tensor (bid : Bid):
    return jnp.concatenate([bid.suit,
                            bid.rank,
                            jax.nn.one_hot(bid.author,4)], axis=1)



# -------------* Utils for Bid manipulation *----------------- #
@jax.jit
def player_pass_on_bid (bid : Bid, player : Int [Array, "B"]) -> Bid:
    return Bid (bid.suit, bid.rank, bid.author, bid.passed | jax.nn.one_hot(player, 4).astype(bool))

@jax.jit
def bid_is_a_pass (bid : Bid) -> Bool [Array, "B"]:
    return jnp.any(bid.rank,axis=-1) == False


# -------------* Utils for History manipulation *----------------- #

def history_modify_at (history : BidHistory,
                       bid : Bid,
                       index : Int [Array, "B"]):
    """ Modifies the bids in a batched of 9xhistory given a batch of positions:
        Sets history_i[j] = bid[i] for j=index[i]
        Warning: sets the current pointer to the specified index
    """
    entries = database_set (history.entries, bid, index)
    return BidHistory(entries, index) 

@jax.jit
def history_play_bid (history : BidHistory,
                      bid : Bid) -> BidHistory:
    """ Main function: plays the bid and modifies the cursor accordingly:
            - a bid with everything set to False => pass
            - otherwise => raise
        TODO : not very clear if the player modifies the current bid or should return an "empty bid". For example, for *pass*:
            - if modified bid = bid with the flag "passed" ON
            - if empty bid = bid with everything at FALSE
    """
    passed_history = history_modify_at(history, bid, history.index)
    raised_history = history_modify_at(history, bid, history.index+1)
    passed_p = bid_is_a_pass(bid)
    return jtu.tree_map(
                lambda passed, raised:
                            jnp.where(passed_p[:,None,None], passed, raised),
                passed_history, raised_history)

@jax.jit
def history_initialize (dealer : Int [Array, "B"]):
    """ initialize a history with dummy empty bids made by the dealer player """
    batch_size = dealer.shape[0]
    history_size = 9 # 9 possible bids max
    entries = Bid (jnp.zeros([batch_size,history_size, 6], dtype=bool),
                   jnp.zeros([batch_size,history_size, 9], dtype=bool),
                   jnp.tile(dealer[:,None],(1,history_size)),
                   jnp.zeros([batch_size,history_size,4], dtype=bool))
    return BidHistory(entries, jnp.zeros([batch_size], dtype=int))
    


def dummy_bid():
   return Bid (jax.nn.one_hot(jnp.array([3,1]),6, dtype=bool),
               jax.nn.one_hot(jnp.array([2,4]), 9, dtype=bool),
               jnp.array([3,1]),
               jnp.zeros([2, 4], dtype=bool))
