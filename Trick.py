from flax import struct
import jax
import jax.numpy as jnp
import jax.random as rnd

from Hand import *



@struct.dataclass
class Trick:
    cards : jnp.array # Players x Color x Rank : bool
    color : jnp.int32 # Color (first color played)
    best_card : jnp.array # Color x Rank : bool
    best_team : jnp.array # Bool (0 for team 1, 1 for team 2)



def  onTrumpTrick (player, hand, trick):
    trump = trick.color
    hastrumpP = hasColor(hand, trump)
    hasbettertrumpP = jnp.any(betterOnColor(trick.best_card)

def possibleMoves (trump, player, hand, trick):
    istrumptrick = trick.color == trump # trumps are asked
    aretrumpplayed = best_card.color == trump # some trump have been played
    hascolorP = hasColor(hand, trick.color) #the player can play this color
    hastrumpP = hasColor(hand, trump) # the player has trumps
    winP = trick.best_team == player % 2 #the team of the player is currently winning the trick

    condHasColor = hascolorP
    condHasToOvertrump = (jnp.logical_not(condHasColor) 
                       & jnp.logical_not(winP) 
                       & aretrumpplayed
                       & hastrumpP)
    condHasToCut = ( jnp.logical_not(condHasColor)
                   & jnp.logical_not(condHasToOvertrump)
                   & jnp.logcial_not(winP))
                  

    colormask = getColorMask(trick.color) 
    trumpmask = (aretrumpplayed*betterOnColor(trick.best_card)+ #allowed trumps
                 (1-aretrumpplayed)*getColorMask(trump))

    


@jax.jit
def newTrick (player, card):
    cards = jnp.zeros([4, 4, 8])
    cards = cards.at[player].set(card)
    return Trick (cards, card.color, card, player % 2)
