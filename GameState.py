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
    return lax.cond(
                trick.suit == trump,
                lambda _: possible_moves_on_trump_trick(trick, player, hand),
                lambda _: possible_moves_on_color_trick(trump, trick, player, hand),
            None)


@jax.jit
def possible_moves_on_trump_trick (trick : Trick, player, hand : Hand) -> Hand:
    trump = trick.suit 
    trumps_in_hand = sh_get_suit(trump, hand)
    has_trumps_p = jnp.any(trumps_in_hand)

    @jax.jit
    def on_has_trump(_):
        higher_trumps_in_hand = sh_higher_in_suit(trick.best_card, hand)
        can_overtrump_p = jnp.any(higher_trumps_in_hand)
        return jnp.bool(can_overtrump_p*higher_trumps_in_hand + (1-can_overtrump_p)*trumps_in_hand)

    return lax.cond(
                has_trumps_p,
                on_has_trump,
                lambda _: hand,
                None)


@jax.jit
def possible_moves_on_color_trick (trump : Suit, trick : Trick, player, hand : Hand) -> Hand:
    suit_in_hand = sh_get_suit(trick.suit, hand)
    has_suit_p = jnp.any(suit_in_hand)


    @jax.jit
    def not_best_team(_):
       """ cannot play the suit and the team loses the trick:
           if someone cut and you can overcut, play this.
           if someone cut and you cannot overcut, play what you want. 
           Otherwise, cut if you have trumps"""
       trick_cut_p = trick.best_card.suit == trump #someone played trump
       overtrumps = sh_higher_in_suit(trick.best_card, hand)
       trumps = sh_get_suit(trump, hand)
       has_trumps_p = jnp.any(trumps)

       return lax.cond(
                      trick_cut_p & jnp.any(overtrumps),
                      lambda _: overtrumps,
                      lambda _: jnp.bool(
                                  trick_cut_p * hand  # cannot play the suit and cannot overtrump => subway rule
                                + jnp.logical_not(trick_cut_p)*(has_trumps_p*trumps + (1-has_trumps_p)*hand)),
                      None)
    @jax.jit    
    def has_no_suit(_):
       """ cannot play the corresponding suit:
           if your team win, you can play whatever""" 
       win_p = (player % 2) == trick.best_team
       return lax.cond(
                      win_p,
                      lambda _: hand,
                      not_best_team,
                      None)
    
    return lax.cond(
                has_suit_p,
                lambda _: jnp.bool(suit_in_hand),
                has_no_suit,
                None)



def test_trick():
    trump = 3
    t = new_trick(0, Card(0,3))
    t = play(trump, t, 1, Card(3,2))
    return t
