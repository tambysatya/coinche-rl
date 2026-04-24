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
#   - 3 policy: for bidding, coinching and overcoinching
# Representations:
#   - A bid is store in the history
#   - A players "passes" a bid by modifying the corresponding flag of the bid
#   - "Coinching" a bid is done by reinitializing the "passing" flags, and setting one "coinching" flag to True
#   - "Overcoinching" a bid is done by setting the corresponding "overcoinching" flag to true
# Total Rollout:
#   - Someone enounce a bidding (can be a "pass")
#   - Coinche Rollout (only taken into consideration if the bidding is not a "pass"):
#               - Each opponent has the opportunity to coinche
#   - Overcoinche Rollout (only taken into consideration if an opponent has coinched):
#               - Each member of the bidding team has the opportunity to overcoinched



@struct.dataclass
class Bid:
    """ A bid made by a player """
    suit : Bool [Array, "B 6"] # one-hot [0,5] total 6: all 4 suits + ALL_TRUMP + NO_TRUMP 
    rank : Bool [Array, "B 10"] # one-hot 0-9 total 10: from PASS (1 call) + 80 to 160 (8 calls) + ALL_IN 

@struct.dataclass
class BidRecord:
    """ A record of a bid made by someone in the past
    """
    bid : Bid
    author : Int [Array, "B"] # index of the player who called this bid
    passed : Bool [Array, "B 4"] # who passed on the bid
    coinched : Bool [Array, "B 4"] # TODO (not implemented) who coinched the bid 
    overcoinched : Bool [Array, "B 4"] # TODO (not implemented) who overcoinched the bid

@struct.dataclass
class BidHistory:
    """ Represents all the previous bid, and a pointer to the current bid:
        Sice ranks can go from 80 to 160 (total=8) + allin (total=9), as most 9 possible bid can be called in a single game.
    """
    entries : BidRecord # B 9 BidRecord
    index : Int [Array, "B"] # index of the current bid

@jax.jit
def record_to_tensor (rec : BidRecord):
    return jnp.concatenate([bid_to_tensor(rec.bid),
                            jax.nn.one_hot(rec.author,4),
                            rec.passed,
                            rec.coinched,
                            rec.overcoinched],axis=-1)

@jax.jit
def history_to_tensor (history : BidHistory):
    entries = history.entries # B 10 BidRecord
    
    def entries_to_tensor (e): # 10 Record
        return jax.vmap(record_to_tensor)(e).flatten()
        

    return jnp.concatenate([jax.vmap(entries_to_tensor)(entries),
                            history.index[:,None]], axis=1)


#--------------------* Predicate on history *----------------------#
# TODO not tested: not used

@jax.jit
def history_is_empty (history : BidHistory) -> Bool [Array, "B"]:
    return history.index == 0
@jax.jit
def history_can_raise (history : BidHistory) -> Bool [Array, "B"]:
    rec = bid_history_current_record(history)
    is_all_in = rec.action.rank[:,-1] 
    is_coinched = rec.coinched.any(axis=-1)
    return is_all_in | is_coinched
@jax.jit
def history_is_bidding_done (history : BidHistory) -> Bool [Array, "B"]:
    rec = bid_history_current_record(history)
    everyone_pass = rec.passed.all(axis=-1)
    overcoinched = rec.overcoinched.any(axis=-1)
    return everyone_pass | overcoinched



# -------------------* Actions on history *------------------------#
@jax.jit
def bid_history_current_record (history : BidHistory) -> BidRecord:
    return database_get(history.entries,history.index)
@jax.jit
def history_player_pass (history : BidHistory,
                         player : Int [Array, "B"]) -> BidHistory:
    """ Modifies the current bid to specify that the specified player passed"""
    rec = bid_history_current_record(history)
    rec = rec.replace(passed = database_set(
                                   rec.passed, jnp.ones_like(player, dtype=bool), player))
    new_entries = database_set(history.entries, rec, history.index)
    return history.replace(entries=new_entries)

@jax.jit
def history_player_bid (history : BidHistory,
                        player : Int [Array, "B"],
                        suit : Int [Array, "B"],
                        rank : Int [Array, "B"]):
    """ Appends the bid made by the specified player to the history """
    batch_size = player.shape[0]
    rec = BidRecord (
               Bid (jax.nn.one_hot(suit,6, dtype=bool),
                     jax.nn.one_hot(rank,10, dtype=bool)),
               player,
               jnp.zeros([batch_size, 4], dtype=bool), # noone passed
               jnp.zeros([batch_size, 4], dtype=bool), # noone coinched
               jnp.zeros([batch_size, 4], dtype=bool)) # noone overcoinched
    return history.replace (index = history.index + 1,
                            entries = database_set(history.entries, rec, history.index+1))


# -------------* Utils for Bid manipulation *----------------- #
@jax.jit
def player_pass_on_bid (rec : BidRecord, player : Int [Array, "B"]) -> Bid:
    return rec.replace(passed = rec.bid.passed | jax.nn.one_hot(player,4).astype(bool))
    #return Bid (bid.suit, bid.rank, bid.author, bid.passed | jax.nn.one_hot(player, 4).astype(bool))

@jax.jit
def bid_is_a_pass (bid : Bid) -> Bool [Array, "B"]:
    return jnp.any(bid.rank,axis=-1) == False


# -------------* Utils for History manipulation *----------------- #

@jax.jit
def history_replace_bid (history : BidHistory,
                         rec : BidRecord):
                       
    """ Modifies the bids in a batched of 9xhistory given a batch of positions:
        Sets history_i[j] = bid[i] for j=index[i]
        Warning: sets the current pointer to the specified index
    """
    entries = database_set (history.entries, rec, index)
    return history.replace(entries=entries)

@jax.jit
def history_initialize (dealer : Int [Array, "B"]):
    """ initialize a history with dummy empty bids made by the dealer player """
    batch_size = dealer.shape[0]
    history_size = 10 # 10 possible bids max
    entries = BidRecord (
                   Bid (jnp.zeros([batch_size,history_size, 6], dtype=bool),
                        jnp.zeros([batch_size,history_size, 10], dtype=bool)),
                   jnp.tile(dealer[:,None],(1,history_size)),
                   jnp.zeros([batch_size,history_size,4], dtype=bool),
                   jnp.zeros([batch_size,history_size,4], dtype=bool),
                   jnp.zeros([batch_size,history_size,4], dtype=bool))
    return BidHistory(entries, jnp.zeros([batch_size], dtype=int))
    


# -------------* Others *-----------------------#
@jax.jit
def bid_to_tensor (bid : Bid):
    return jnp.concatenate([bid.suit,
                            bid.rank],axis=-1)

