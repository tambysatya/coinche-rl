from flax import struct
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as rnd

from Card import *
from Hand import *



@struct.dataclass
class Trick:
    suit : Suit  # Color (first color played)
    best_card : Card # Color x Rank : bool
    best_team : jnp.int32 # (0 for team 1, 1 for team 2)
    cards : Bool [Array, "4 4 8"] # Player x TensorCard



def play (trump : Suit,
          trick : Trick,
          player,
          card : Card) -> Trick:
    """ Inserts a new card in the Trick """
    same_best_p = is_better_p(trump, trick.best_card, card)
    tensorcard = card_to_tensor(card)
    cards = trick.cards.at[player].set(tensorcard)

    return lax.cond(same_best_p,
                    lambda _: Trick(trick.suit, trick.best_card, trick.best_team, cards),
                    lambda _: Trick(trick.suit, card, player % 2, cards),
                    None)


def new_trick (player, first_card : Card):
    tensor = card_to_tensor(first_card)
    cards = jnp.zeros([4,4,8], dtype=bool)
    cards = cards.at[player].set(tensor)
    return Trick (first_card.suit, first_card, player % 2, cards)

