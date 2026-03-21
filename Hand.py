from flax import struct
import jax
import jax.numpy as jnp
import jax.random as rnd
from jaxtyping import Array, Bool

from Card import *

ACE, TEN, K, Q, J, NINE, EIGHT, SEVEN = 0,1,2,3,4,5,6,7
to_trump = jnp.array([J,NINE,ACE,TEN,K,Q,EIGHT,SEVEN])


key = rnd.key(0)

  
Hand = Bool [Array, "4 8"]

@jax.jit
def set_trump (suit : Suit, hand : Hand) -> Hand:
        """ Reorders the specified suit in the hand """
        row = hand[suit][to_trump]
        newhand = hand.at[suit].set(row)
        return newhand


#Accessors (subhands)

@jax.jit
def sh_get_suit (suit : Suit, hand : Hand) -> Hand:
    """ Returns the subhand of cards of the suit """
    return hand*jax.nn.one_hot(suit,4)[:,None]

@jax.jit
def sh_higher_in_suit (card : Card, hand : Hand) -> Hand:
    """ Returns the subhand of cards better than card """
    mask = jnp.zeros_like(hand, dtype=bool)
    ranks = jnp.arange(8) < card.rank
    return hand * mask.at[card.suit].set(ranks)


def randomHand ():
    h = rnd.uniform(key, [4,8])
    return jnp.bool(jnp.where(h > 0.5, 1, 0))

