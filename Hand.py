from flax import struct
import jax
import jax.numpy as jnp
import jax.random as rnd
from jaxtyping import Array, Bool

from Card import *

ACE, TEN, K, Q, J, NINE, EIGHT, SEVEN = 0,1,2,3,4,5,6,7
to_trump = jnp.array([J,NINE,ACE,TEN,K,Q,EIGHT,SEVEN])


key = rnd.key(0)

  
Hand = Bool [Array, "B 4 8"]

@jax.jit
def set_trump (suits : Suit, hands : Hand) -> Hand:
    """ Reorders the specified suit in the hand """
    def scalar_set_trump (suit : Suit, hand : Hand) -> Hand:
            row = hand[suit][to_trump]
            newhand = hand.at[suit].set(row)
            return newhand
    return jax.vmap(scalar_set_trump)(suits, hands)


#Accessors (subhands)

@jax.jit
def sh_get_suit (suits : Suit, hands : Hand) -> Hand:
    """ Returns the subhand of cards of the suit """
    def scalar_sh_get_suit (suit : Suit, hand : Hand) -> Hand:
        return hand*jax.nn.one_hot(suit,4, dtype=bool)[:,None]
    return jax.vmap(scalar_sh_get_suit)(suits, hands)

@jax.jit
def sh_higher_in_suit (cards : Card, hands : Hand) -> Hand:
    """ Returns the subhand of cards better than card """
    def scalar_sh_higher_in_suit (card : Card, hand : Hand) -> Hand:
        mask = jnp.zeros_like(hand, dtype=bool)
        ranks = jnp.arange(8) < card.rank
        return hand * mask.at[card.suit].set(ranks)
    return jax.vmap(scalar_sh_higher_in_suit)(cards, hands)


def randomHand (key=key):
    h = rnd.uniform(key, [4,8])
    return jnp.bool(jnp.where(h > 0.5, 1, 0))

def randomHands(keys):
    n_key = keys.shape[0]
    h = rnd.uniform(key, [n_key, 4,8])
    return jnp.bool(jnp.where(h > 0.5, 1, 0))


