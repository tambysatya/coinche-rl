from flax import struct
import jax
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



def play (trump : Suit, trick : Trick, player, card : Card):
    pass

def new_trick (player, first_card : Card):
    tensor = card_to_tensor(first_card)
    cards = jnp.zeros([4,4,8], dtype=bool)
    cards = cards.at[player].set(tensor)
    return Trick (first_card.suit, first_card, player % 2, cards)

