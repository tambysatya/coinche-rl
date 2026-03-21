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
    best_team : Bool [Array, "B"] # (0 for team 1, 1 for team 2)
    cards : Bool [Array, "B 4 4 8"] # Player x TensorCard



def play (trumps : Suit,
          tricks : Trick,
          players,
          cards : Card) -> Trick:
    def scalar_play (trump : Suit,trick : Trick,player,card : Card) -> Trick:
        """ Inserts a new card in the Trick """
        same_best_p = is_better_p(trump, trick.best_card, card)
        tensorcard = card_to_tensor(card)
        cards = trick.cards.at[player].set(tensorcard)
        return lax.cond(same_best_p,
                        lambda _: Trick(trick.suit, trick.best_card, trick.best_team, cards),
                        lambda _: Trick(trick.suit, card, (player % 2).astype(jnp.bool), cards),
                        None)
    return jax.vmap(scalar_play)(trumps, tricks, players, cards)


@jax.jit
def new_trick(players, first_cards : Card) -> Trick:
    batch_size = players.shape[0]
    tensor = card_to_tensor(first_cards)
    cards = jnp.zeros([batch_size, 4,4,8], dtype=bool)

    def initialize(cards, player, tensor, first_cards):
        cards = cards.at[player].set(tensor)
        return Trick (first_cards.suit, first_cards, players % 2, cards)
    return jax.vmap(initialize)(cards, players, tensor, first_cards)

