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
    trump : jnp.int32


#Reorders the colors (if it was sans at, it becomes trump, and vice versa) 
@jax.jit
def getTrump (hand : Hand, color) -> jax.Array:
        return hand.cards[color][to_trump]

@jax.jit
def getColor (hand : Hand, color) -> jax.Array:
        return hand.cards[color]


@jax.jit
def isTrump(hand : Hand, color) -> jnp.bool:
    return (hand.trump == color) | (hand.trump == ALL_TRUMP)

@jax.jit
def canCut (hand:Hand) -> jnp.bool:
    """ True if cutting is allowed """
    return (hand.trump != ALL_TRUMP) & (hand.trump != NO_TRUMP)

@jax.jit
def getCards (hand : Hand, color) -> jax.Array:
    """
        Returns the boolean vectors of cards corresponding to the specified color.
        The cards are sorted in decreasing order, i.e. the index 0 correspond to ACE or JACK depending on the type of the color (TRUMP or COLOR).
    """
    return jax.lax.cond(
                isTrump(hand, color),
                lambda _: getTrump(hand, color),
                lambda _: getColor(hand, color),
                operand=None
            )


def showHand (hand : Hand) -> str :
        lines = []
        for color in range(4):
            cards = getCards(hand, color)
            lines.append(str(jnp.nonzero(cards)))

        return "\n".join(lines)



def random_hand():
    hand = jnp.where(rnd.uniform(key,[4,8]) > 0.5, 1, 0)
    return Hand(hand, 0)
