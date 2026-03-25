from flax import struct
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as rnd

from utils import *
from coinche.Card import *
from coinche.Hand import *


TensorTrick = Bool [Array, "B 69"] # 4 + 12 + 4 + (4*12) + 1

@struct.dataclass
class Trick:
    suit : Suit  # Suit [INDEX] (first suit played)
    best_card : Card # [B Cards]: best card played so far
    best_player : Int [Array, "B"] # best player so far (known as master player) : Int
    cards : Bool [Array, "B 4 12"] # Player x TensorCard : [B x 4  x 12]
    startedP : Bool [Array, "B"] #True if the trick has started

def trick_to_tensor (trick: Trick) -> TensorTrick:
    return jnp.concatenate([jax.nn.one_hot(trick.suit, 4),
                           card_to_tensor(trick.best_card).reshape(-1,12),
                           jax.nn.one_hot(trick.best_player,4),
                           trick.cards.reshape(-1, 4*12),
                           trick.startedP.reshape(-1,1)],
                           axis=1)    

def play (trumps : Suit,
          tricks : Trick,
          players,
          cards : Card) -> Trick:

    """ Inserts the card played by a player into the trick. Providing the trump suit is mandatory
        to identifies which player is winning the trick so far"""

    best_card = tricks.best_card
    same_best_p = is_better_p(trumps, best_card, cards)
    startedP = tricks.startedP
    tensorcards = card_to_tensor(cards)

    def insert_card(player, tensorcard, trickcard):
        return trickcard.at[player].set(tensorcard)

    new_trick_cards = jax.vmap(insert_card)(players, tensorcards, tricks.cards)
    new_best_players = jnp.where(same_best_p, tricks.best_player, players)

    new_trick_suits = jnp.where(startedP, tricks.suit, cards.suit)
    new_best_cards_suits = jnp.where(same_best_p, best_card.suit, cards.suit)
    new_best_cards_ranks = jnp.where(same_best_p, best_card.rank, cards.rank)
    new_best_cards = Card(new_best_cards_suits, new_best_cards_ranks)


    return Trick(new_trick_suits, new_best_cards, new_best_players, new_trick_cards, jnp.ones_like(players, dtype=bool))


@jax.jit
def new_trick(players) -> Trick:
    """ Generates an empty trick """
    batch_size = players.shape[0]
    dummy_suit = jnp.zeros([batch_size], dtype=int)
    dummy_best_card = Card(jnp.zeros ([batch_size], dtype=int), jnp.zeros([batch_size], dtype=int))
    cards = jnp.zeros([batch_size, 4,12], dtype=bool)
    startedP = jnp.zeros_like(players, dtype=bool)

    return Trick(dummy_suit, dummy_best_card, players, cards, startedP)

    

def show_trick(trump, trick: Trick, index=0) -> str:
    """ Displays a trick from a batch (default = 0)."""
    suit = trick.suit[index]
    cards = trick.cards[index] 
    best_card = Card(trick.best_card.suit[index], trick.best_card.rank[index])
    best_player = trick.best_player[index]
    startedP = trick.startedP[index]

    if not startedP:
        return "[]"
    ret = [f"[best={best_player}]"]
    for i, card_ in enumerate(cards):
        if jnp.any(card_):
            card = card_from_tensor(card_.reshape(1,-1))
            if card == best_card:
                ret.append(f"{bcolors.BOLD}{bcolors.UNDERLINE}{show_card(trump, card, i)}{bcolors.ENDC}")
            else:
                ret.append(f"{show_card(trump, card, i)}")
        else:
            ret.append("?")
    return " ".join(ret)
        

    

def mk_setup():
    players = jnp.array([1])
    cards = Card(jnp.array([2]), jnp.array([1]))
    trumps = jnp.array([1])
    return players, cards, trumps
