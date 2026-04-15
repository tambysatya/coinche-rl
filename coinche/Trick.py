from flax import struct
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as rnd
import jax.tree_util as jtu
from jaxtyping import Float

from utils import *
from coinche.Card import *
from coinche.Hand import *
from coinche.Bid import *


Player = Int [Array, "B"]

@struct.dataclass
class Trick:
    """ Describes a (partially played) trick """
    startedP : Bool [Array, "B"] #True if the trick has started
    suit : Suit  # [B Suit] (first suit played)
    best_card : Card # [B Cards]: best card played so far
    best_player : Player # best player so far (known as master player) : Int

    value : Float [Array, "B"] # current value of the trick
    cards : Card  # Cards already played in the trick:  Player x Card : [B x 4]
    hands : Hand # Current hand of each player: [B, 4, Hands] = [B, 4, 4 , 8]
    current_player : Player # The player who has to play this turn
    size : Int [Array, "B"] # current size of the trick [0..4]

@struct.dataclass
class TrickHistory:
    """ This encodes the entire trick phase until now """
    trump : Suit
    entities : Trick # B 8 Trick
    index : Int [Array, "B"] # Position of the curent trick



@jax.jit
def trick_obs(trick: Trick):
    """ Observation of a trick where the only visible hand is the one of the current player """
    player = trick.current_player
    batch_size = player.shape[0]

    player_hand = jax.vmap(lambda hand, p: hand[p])(trick.hands, player)
    trick_cards = jax.vmap(lambda c: card_to_tensor(c).reshape([1,-1]))(trick.cards)
    trick_cards = trick_cards.reshape([batch_size, -1])
    
    return jnp.concatenate([trick.suit[:,None],
                            card_to_tensor(trick.best_card),
                            trick.best_player[:,None],
                            trick.value[:,None],
                            trick_cards,
                            player_hand.reshape([-1, 4*8]),
                            trick.current_player[:,None],
                            trick.size[:,None]], axis=1)
    


def trick_history_initialize (history : BidHistory, hands : Hand):
   """ Setups an empty game, given a batch of history and a batch of 4 hands """
   rec = history_current_record (history)
   initial_players, best_suit = rec.author, rec.bid.suit

   batch_size = rec.author.shape[0]

   #initialize 8 empty tricks per batch
   tricks = new_trick(initial_players, hands)
   tricks = jtu.tree_map(
                lambda l: jnp.tile(l, (1,)+(8,) + (1,)*(l.ndim-1)),
                tricks)

   return tricks




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
                 new_value, new_cards, new_hands, current_player, tricks.size + 1)

    
@jax.jit
def new_trick(initial_players, hands : Hand) -> Trick:
    """ Generates an empty trick:
        The current hands of each player must be provided: [B, 4, 4 , 8] = [B 4 Hands]"""
    batch_size = initial_players.shape[0]
    startedP = jnp.zeros_like(initial_players, dtype=bool)
    dummy_suit = jnp.zeros_like(initial_players, dtype=int)
    dummy_best_card = Card(jnp.full(batch_size,-1, dtype=int), jnp.full(batch_size,-1, dtype=int))

    value = jnp.zeros_like(initial_players, dtype=float)
    cards = Card(jnp.full([batch_size, 4],-1, dtype=int), jnp.full([batch_size, 4],-1, dtype=int))

    return Trick(startedP, dummy_suit, dummy_best_card, initial_players, value, cards, hands, initial_players, jnp.zeros_like(initial_players, dtype=int))



def show_trick(trump, trick: Trick, index=0) -> str:
    """ Displays a trick from a batch (default = 0)."""
    suit = trick.suit[index]
    cards = jtu.tree_map(lambda t: t[index], trick.cards)
    best_card = Card(trick.best_card.suit[index], trick.best_card.rank[index])
    best_player = trick.best_player[index]
    startedP = trick.startedP[index]

    if not startedP:
        return "[]"
    ret = [f"[best={best_player}]"]
    for i, card_ in enumerate(cards):
        if card_.suit >= 0 and card.rank >= 0:
            if card == best_card:
                ret.append(f"{bcolors.BOLD}{bcolors.UNDERLINE}{show_card(trump, card, i)}{bcolors.ENDC}")
            else:
                ret.append(f"{show_card(trump, card, i)}")
        else:
            ret.append("?")
    return " ".join(ret)
        


