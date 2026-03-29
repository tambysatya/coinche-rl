from flax import struct
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as rnd
import jax.tree_util as jtu

from utils import *
from coinche.Card import *
from coinche.Hand import *


Player = Int [Array, "B"]

@struct.dataclass
class Trick:
    """ Describes a (partially played) trick """
    startedP : Bool [Array, "B"] #True if the trick has started
    suit : Suit  # Suit [INDEX] (first suit played)
    best_card : Card # [B Cards]: best card played so far
    best_player : Player # best player so far (known as master player) : Int

    value : Int [Array, "B"] # current value of the trick
    cards : Card  # Cards already played in the trick:  Player x Card : [B x 4]
    hands : Hand # Current hand of each player: [B, 4, Hands] = [B, 4, 4 , 8]
    current_player : Player




@jax.jit
def play (trumps : Suit,
          tricks : Trick,
          cards : Card) -> Trick:

    """ Inserts the card played by a player into the trick. Providing the trump suit is mandatory
        to identifies which player is winning the trick so far"""

    players = tricks.current_player

    best_card = tricks.best_card
    same_best_p = is_better_p(trumps, best_card, cards)
    startedP = tricks.startedP

    # inserts the card in trick (for an entry in the batch)
    def insert_card (old_cards, player, card):
        return jtu.tree_map(lambda array, val: array.at[player].set(val), old_cards, card)
    new_cards = jax.vmap(insert_card)(tricks.cards, players, cards)

    # removes the card in the hand of the player (for an entry in the batch)
    def remove_card (player_hand, card):
        return player_hand * (~card_to_subhand(card))
    new_hands = jax.vmap (lambda hand,p, card: hand.at[p].set(remove_card(hand[p], card)))(tricks.hands, players[:,None],cards) #vmap over the batch

    new_best_players = jnp.where(same_best_p, tricks.best_player, players)

    new_trick_suits = jnp.where(startedP, tricks.suit, cards.suit)
    new_best_cards = jtu.tree_map(lambda old,new: jnp.where(same_best_p,old, new), best_card, cards)
    new_value = tricks.value + card_value(trumps, cards)
    current_player = (tricks.current_player + 1) % 4


    return Trick(jnp.ones_like(players, dtype=bool),
                 new_trick_suits, new_best_cards, new_best_players,
                 new_value, new_cards, new_hands, current_player)

    

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
        

@jax.jit
def new_trick(initial_players, hands : Hand) -> Trick:
    """ Generates an empty trick:
        The current hands of each player must be provided: [B, 4, 4 , 8] = [B 4 Hands]"""
    batch_size = initial_players.shape[0]
    startedP = jnp.zeros_like(initial_players, dtype=bool)
    dummy_suit = jnp.zeros_like(initial_players, dtype=int)
    dummy_best_card = Card(jnp.full(batch_size,-1, dtype=int), jnp.full(batch_size,-1, dtype=int))

    value = jnp.zeros_like(initial_players, dtype=int)
    cards = Card(jnp.full([batch_size, 4],-1, dtype=int), jnp.full([batch_size, 4],-1, dtype=int))

    return Trick(startedP, dummy_suit, dummy_best_card, initial_players, value, cards, hands, initial_players)


    

def mk_setup():
    players = jnp.array([1])
    cards = Card(jnp.array([2]), jnp.array([1]))
    trumps = jnp.array([1])
    return players, cards, trumps
