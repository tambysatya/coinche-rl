from flax import struct
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jax.random as rnd
import jax.tree_util as jtu
from jaxtyping import Array, Bool, Int, Float

from utils import *
from coinche.Hands import *

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
    """ A bid made by a player """
    suit : Int [Array, "B"] # [0,5] total 6: all 4 suits + ALL_TRUMP + NO_TRUMP 
    rank : Int [Array, "B"] #0-8 total 9: from 80 to 160 (8 calls) + ALL_IN 


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
    turn :: Int [Array, "B"]  # index of the current turn
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
    return jnp.concatenate([jnp.one_hot(bid.suit,6),
                            jnp.one_hot(bid.rank, 9)], axis=1)

def mk_coinche_rollout(coinche_mdl, pool_size):
    graphdef, _ = nnx.split(coinche_mdl)

    def coinche_p (player_coinche_param,
                   player_hand : Hand,
                   history : BidEntry,
                   bid : Bid,
                   key):
        coinche_actor = nnx.merge(graphdef, player_coinche_param)
        obs = CoincheObs(player_hand, history, bid)
        logit = coinche_actor(obs)
        action = rnd.categorical(key, jnp.array([1-logit, logit]))
        return action, Step(obs, action, jnp.log(logit))


    def coinche_rollout (all_coinche_params,
                         all_overcoinche_params,
                         permutation,
                         bidding_player : Int [Array, "B"], # the player who announced
                         hand : Hand , # Hand of all the 4 players
                         history : BidEntry,  #previous bidding
                         bid : Bid,
                         seed) -> BidEntry:
        """ Asks the other teams if they coinche """
        
        first_player = (1+bidding_player) % 2
        second_player = (first_player + 2) % 2

        defense_team_permutation = permutation[first_player % 2]
        bidding_team_permutation = permutation[bidding_player % 2]


        defense_first_hand, defense_second_hand = jax.vmap(lambda h, f, s = h[f], h[s])(hand, first_player, second_player)
        defense_first_hand, defense_second_hand, history, bid = group_dataset_by_agent(pool_size, defense_team_permutation, (defense_first_hand, defense_second_hand, history, bid))

        def1, def2, bid1, bid2 = rnd.split(seed, 4)
        defense_first_coinche = jax.vmap(coinche_p)(all_coinche_params, defense_first_hand, history, bid, def1)
        defense_snd_coinche = jax.vmap(coinche_p)(all_coinche_params, defense_second_hand, history, bid, def2)

        bidding_first_hand, bidding_second_hand = jax.vmap(lambda h, f, s = h[f], h[s])(hand, current_player, (current_player + 2) %2)
        bidding_first_hand, bidding_second_hand, history, bid = group_dataset_by_agent(pool_size, bidding_team_permutation, (bidding_first_hand, bidding_second_hand, history, bid))
        
                         

