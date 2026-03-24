from flax import struct
import jax
import jax.numpy as jnp
import jax.random as rnd
from jaxtyping import Array, Bool, Int

  

Suit = Int [Array, "B"]
Rank = Int [Array, "B"]
TensorSuit = Bool [Array, "B 4"] # One-hot encoding of a suit
TensorCard = Bool [Array, "B 12"] #Vectorized representation of a card SUIT(4) + RANK (8)

@struct.dataclass
class Card:
    suit : Suit # [B Int]
    rank : Rank # [B Int]

@jax.jit
def card_to_tensor(card : Card) -> TensorCard:
    suits = jax.nn.one_hot(card.suit, 4, dtype=bool)
    ranks = jax.nn.one_hot(card.rank, 8, dtype=bool)
    return jnp.concatenate([suits, ranks], axis=-1)

@jax.jit
def card_from_tensor(ts : TensorCard) -> Card:
    suits, ranks = ts[:, :4], ts[:, 4:]
    return Card(suits.argmax(axis=1), ranks.argmax(axis=-1))


def show_card(is_trump : bool, card : Card, index=0) -> str:
    suit = ["♠", "♥", "♦", "♣"]
    rank = ["A", "10", "K", "Q", "J", "9", "8", "7"]
    trump_rank = ["J", "9", "A", "10", "K", "Q", "8", "7"]

    card_suit = card.suit[index].item()
    card_rank = card.rank[index].item()

    if is_trump:
        card_rank = trump_rank[card_rank]
    else:
        card_rank = rank[card_rank]
    return card_rank + suit[card_suit]


@jax.jit
def is_better_p (trump : Suit, cardA : Card, cardB : Card) -> jnp.bool:
   """ True if cardA is better than cardB . Note: cardA is assumed to belong to the leading suit """ 
   same_color_p = cardA.suit == cardB.suit
   cardb_trump_p = trump == cardB.suit
   return jnp.bool(same_color_p*(cardA.rank < cardB.rank) | jnp.logical_not(cardb_trump_p))

 
