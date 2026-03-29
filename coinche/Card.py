from flax import struct
import jax
import jax.numpy as jnp
import jax.random as rnd
from jaxtyping import Array, Bool, Int

from utils import *
  

Suit = Int [Array, "B"]
Rank = Int [Array, "B"]
TensorSuit = Bool [Array, "B 4"] # One-hot encoding of a suit
TensorCard = Bool [Array, "B 12"] #Vectorized representation of a card SUIT(4) + RANK (8)

suit_values = jnp.array([11,10,4,3,2,0,0,0])
trump_values = jnp.array([20,14,11,10,4,3,2,0,0])*2/3 #TODO
no_trump_values = jnp.array([20,10,4,3,2,0,0,0])

SUIT_ALL_TRUMP=5
SUIT_NO_TRUMP=6

@struct.dataclass
class Card:
    suit : Suit # [B Int]
    rank : Rank # [B Int]

@jax.jit
def card_to_tensor(card : Card) -> TensorCard:
    suits = jax.nn.one_hot(card.suit, 4, dtype=bool)
    ranks = jax.nn.one_hot(card.rank, 8, dtype=bool)
    return jnp.concatenate([suits, ranks], axis=-1)

def dummy_card() -> Card:
    """ Creates a dummy card (encoded as [False....False])
        Aims to describe an unknown card"""
    return Card(jnp.array([-1])[:,None], jnp.array([-1])[:,None])

@jax.jit
def card_from_tensor(ts : TensorCard) -> Card:
    suits, ranks = ts[:, :4], ts[:, 4:]
    return Card(suits.argmax(axis=1), ranks.argmax(axis=-1))

@jax.jit
def card_to_subhand (card : Card) -> Bool [Array, "B 4 8"]:
    idx = card.suit*8 + card.rank
    return jax.nn.one_hot(idx,32, dtype=bool).reshape(-1, 4, 8)


def show_card(trump, card : Card, index=0) -> str:
    """ Displays the index^th card of the batch (blue if in trump suit) """
    suit = ["♠", "♥", "♦", "♣"]
    rank = ["A", "10", "K", "Q", "J", "9", "8", "7"]
    trump_rank = ["J", "9", "A", "10", "K", "Q", "8", "7"]

    card_suit = card.suit[index].item()
    card_rank = card.rank[index].item()

    if card_suit == trump:
        card_rank = bcolors.OKBLUE + trump_rank[card_rank]
    else:
        card_rank = rank[card_rank]
    return card_rank + suit[card_suit] + bcolors.ENDC

@jax.jit
def card_value (trump : Suit, card : Card) -> str:
    """ Crashes if the card is invalid """
    return jnp.where((trump == card.suit) | (trump == SUIT_ALL_TRUMP),
                     trump_values[card.rank],
                     jnp.where(trump == SUIT_NO_TRUMP,
                               no_trump_values[card.rank],
                               suit_values[card.rank]))

@jax.jit
def is_better_p (trump : Suit, cardA : Card, cardB : Card) -> jnp.bool:
   """ True if cardA is better than cardB . Note: cardA is assumed to belong to the leading suit """ 
   same_color_p = cardA.suit == cardB.suit
   cardb_trump_p = trump == cardB.suit
   return jnp.bool(same_color_p*(cardA.rank < cardB.rank) | jnp.logical_not(cardb_trump_p))


 
