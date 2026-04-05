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
    rank : Bool [Array, "B 9"] #one-hot 0-8 total 9: from 80 to 160 (8 calls) + ALL_IN 
    author : Int [Array, "B"] # index of the player who called this bid


@struct.dataclass
class BidMask:
    """ Masks on the possible bids allowed to a player """
    possible_rank : Bool [Array, "B 6"]
    possible_check : Bool [Array, "B"]
    possible_coinche : Bool [Array, "B"]
    possible_overcoinche : Bool [Array, "B"]


@struct.dataclass
class BidEntry:
    """ A bid that have been called, as well as which players coinched it """
    author : Int [Array, "B"] # The player who made the bidding
    bid : Bid
    coinched : Bool [Array, "B 4"] # The player who coinched the bidding (one-hot, can be all FALSE)
    overcoinched : Bool [Array, "B 4"] # The player who overcoinched the bidding (one-hot, can be all FALSE)


@struct.dataclass
class BidState:
    """ State of the game during the bidding phase """
    turn : Int [Array, "B"]  # index of the current turn
    current_player : Int [Array, "B"]
    best_bid : Bid
    best_player : Int [Array, "B"]
    history : BidEntry 


@struct.dataclass
class CoincheObs:
    hand : Hand 
    history : BidEntry
    bid : Bid

@struct.dataclass
class CoincheStep:
    """ Step of the coinche rollout """
    obs : CoincheObs
    action : Bool [Array, "B"]
    logprob : Float [Array, "B"]


def history_to_tensor(history : BidEntry):
    return jnp.concatenate ([
                       jax.nn.one_hot(history.author, 4),
                       bid_to_tensor(history.bid),
                       coinched,
                       overcoinched
                    ], axis=1)

def bid_to_tensor (bid : Bid):
    return jnp.concatenate([bid.suit,
                            bid.rank], axis=1)

