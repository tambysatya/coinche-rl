from flax import struct
import jax
import jax.numpy as jnp
import jax.random as rnd
from jaxtyping import Array, Bool

  

Suit = jnp.int32
Rank = jnp.int32
TensorCard = Bool [Array, "4 8"] #Vectorized representation of a card

@struct.dataclass
class Card:
    suit : Suit
    rank : Rank

@jax.jit
def card_to_tensor(card : Card) -> TensorCard:
    t = jnp.zeros([4,8], dtype=bool)
    return t.at[card.suit, card.rank].set(True)

@jax.jit
def card_from_tensor(t : TensorCard) -> Card:
    suit, rank = jnp.unravel_index(jnp.argmax(t), t.shape)
    return Card(suit, rank)


def is_better_p (trump : Suit, cardA : Card, cardB : Card) -> jnp.bool:
   """ True if cardA is better than cardB . Note: cardA is assumed to belong to the leading suit """ 
   same_color_p = cardA.suit == cardB.suit
   cardb_trump_p = trump == cardB.suit
   return jnp.bool(same_color_p*(cardA.rank < cardB.rank) | jnp.logical_not(cardb_trump_p))

 
