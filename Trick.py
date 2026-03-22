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



@jax.jit
def play (trumps : Suit,
          tricks : Trick,
          players,
          cards : Card) -> Trick:

    same_best_p = is_better_p(trumps, tricks.best_card, cards)
    tensorcards = card_to_tensor(cards)

    def insert_card(player, tensorcard, trickcard):
        return trickcard.at[player].set(tensorcard)

    new_trick_cards = jax.vmap(insert_card)(players, tensorcards, tricks.cards)
    new_best_teams = jnp.where(same_best_p, tricks.best_team, players % 2)

    new_best_cards_suits = jnp.where(same_best_p, tricks.best_card.suit, cards.suit)
    new_best_cards_ranks = jnp.where(same_best_p, tricks.best_card.rank, cards.rank)
    new_best_cards = Card(new_best_cards_suits, new_best_cards_ranks)


    return Trick(tricks.suit, new_best_cards, new_best_teams, new_trick_cards)


@jax.jit
def new_trick(players, first_cards : Card) -> Trick:
    batch_size = players.shape[0]
    tensor = card_to_tensor(first_cards)
    cards = jnp.zeros([batch_size, 4,4,8], dtype=bool)

    def initialize(cards, player, tensor, first_cards):
        cards = cards.at[player].set(tensor)
        return Trick (first_cards.suit, first_cards, players % 2, cards)
    return jax.vmap(initialize)(cards, players, tensor, first_cards)

