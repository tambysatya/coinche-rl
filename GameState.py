from flax import struct
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as rnd

from Card import *
from Hand import *
from Trick import *



@jax.jit
def possible_moves (trump,
                    trick : Trick,
                    player,
                    hand : Hand) -> Hand:
    trump_trick_p = trick.suit == trump
    return (trump_trick_p * possible_moves_on_trump_trick(trick, player, hand) +
           (1-trump_trick_p)*possible_moves_on_color_trick(trump, trick, player, hand))

@jax.jit
def possible_moves_on_trump_trick (trick : Trick, player, hand : Hand) -> Hand:
    trump = trick.suit 
    trumps_in_hand = sh_get_suit(trump, hand)
    has_trumps_p = jnp.any(trumps_in_hand, axis=(1,2))

    @jax.jit
    def on_has_trump():
        higher_trumps_in_hand = sh_higher_in_suit(trick.best_card, hand)
        can_overtrump_p = jnp.any(higher_trumps_in_hand, axis=(1,2))
        return jnp.bool(can_overtrump_p*higher_trumps_in_hand + (1-can_overtrump_p)*trumps_in_hand)

    return has_trumps_p*on_has_trump() + (1-has_trumps_p)*hand


@jax.jit
def possible_moves_on_color_trick (trump : Suit, trick : Trick, player, hand : Hand) -> Hand:
    suit_in_hand = sh_get_suit(trick.suit, hand)
    has_suit_p = jnp.any(suit_in_hand, axis=(1,2))


    @jax.jit
    def not_best_team():
       """ cannot play the suit and the team loses the trick:
           if someone cut and you can overcut, play this.
           if someone cut and you cannot overcut, play what you want. 
           Otherwise, cut if you have trumps"""
       trick_cut_p = trick.best_card.suit == trump #someone played trump
       overtrumps = sh_higher_in_suit(trick.best_card, hand)
       trumps = sh_get_suit(trump, hand)
       has_trumps_p = jnp.any(trumps, axis=(1,2))

       has_to_ovetrump_p = trick_cut_p & jnp.any(overtrumps, axis=(1,2))
       return (has_to_ovetrump_p* overtrumps +
               (1-has_to_ovetrump_p)*(
                                      trick_cut_p * hand  # cannot play the suit and cannot overtrump => subway rule
                                    + jnp.logical_not(trick_cut_p)*(has_trumps_p*trumps + (1-has_trumps_p)*hand))
              )
    @jax.jit    
    def has_no_suit():
       """ cannot play the corresponding suit:
           if your team win, you can play whatever""" 
       win_p = (player % 2) == trick.best_team
       return win_p*hand + (1-win_p)*not_best_team()
    
    return has_suit_p*suit_in_hand + (1-has_suit_p)*has_no_suit()



def test_hands():
    """ a batch with a single hand """
    subkey = rnd.split(key, 1)
    h = randomHand(key)
    return jnp.expand_dims(h, axis=0)
def test_cards (suit, rank):
    """a batch with a single card """
    return Card(jnp.expand_dims(suit,axis=0),
                jnp.expand_dims(rank, axis=0))

def test_trick(trumps, cardA, cardB):
    """ A (batched) trick containing two cards """
    players0 = jnp.zeros_like(cardA.suit) 
    players1 = jnp.ones_like(cardA.suit) 
    t = new_trick(players0, cardA)
    t = play(trumps, t, players1, cardB)
    return t
