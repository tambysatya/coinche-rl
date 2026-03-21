from flax import struct
import jax
import jax.numpy as jnp
import jax.random as rnd

ACE, TEN, K, Q, J, NINE, EIGHT, SEVEN = 0,1,2,3,4,5,6,7
to_trump = jnp.array([J,NINE,ACE,TEN,K,Q,EIGHT,SEVEN])


key = rnd.key(0)



# Special color indices
ALL_TRUMP=-1
NO_TRUMP=-2

   
@struct.dataclass
class Hand:
    cards : jax.Array #hand: boolean matrix 4x8 ordered in the color order

@struct.dataclass
class Card:
    color : jnp.int32
    rank : jnp.int32


#Reorders the colors (if it was sans at, it becomes trump, and vice versa) 
@jax.jit
def setTrump (hand : Hand, color) -> Hand:
        row = hand.cards[color][to_trump]
        newhand = hand.cards.at[color].set(row)
        return Hand(newhand)


def getColorMask (color) -> jax.Array:
    """
        Keeps only the row corresponding to the specified color
    """
    mask = jnp.ones([4,8])
    color = jax.nn.one_hot(color, 4)
    return color[:,None]*mask



def betterOnColor (card: Card):
   """
        Mask that keeps only the cards of the color above the specified card
   """
   mask = jnp.zeros([4,8])
   cardrank, cardcolor = card.rank, card.color
   mask = mask.at[cardcolor,:cardrank].set(1)
   return mask


#predicates


def hasColor(hand : Hand, color) -> jnp.bool:
    return jnp.any(hand.cards[color])

def canPlayAbove(hand : Hand, color, rank) -> jnp.bool:
    return jnp.any(hand[color,:rank])



def mkHand():
    hand = rnd.uniform(key, [4,8])
    hand = jnp.where(hand > 0.5, 1, 0)
    hand = Hand(hand)
    return setTrump(hand, 0)
