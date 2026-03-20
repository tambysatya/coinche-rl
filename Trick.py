from flax import struct
import jax
import jax.numpy as jnp
import jax.random as rnd

from Hand import *



@struct.dataclass
class Card:
    color: jnp.int32 # 0-3
    rank: jnp.int32 # 0-7

@struct.dataclass
class Trick:
    cards: list[Card] #[Cards]
    first_color: jnp.int32 # color type
    best_card: Card
    winner: jnp.int32 # 0-3: player index


@jax.jit
def newTrick(player_index, first_card):
    return Trick([first_card], first_card.color, first_card, player_index)


def sameTeam (player1: jnp.int32, player2: jnp.int32):
    return (((player1 == 0) & (player2 == 2)) | 
             ((player1 == 2) & (player2 == 0)) | 
             ((player1 == 1) & (player2 == 3)) | 
             ((player1 == 3) & (player2 == 1)))

def isWinning (trick: Trick, player: jnp.int32) -> jnp.bool:
    return sameTeam(player, trick.winner)

def hasColor(trick: Trick, hand: Hand) -> jnp.bool:
    return jnp.any(hand[trick.first_color])

# These functions returns a boolean mask 4x8
def trumpsAbove(rank: jnp.int32, hand:Hand) -> jax.Array:
    mask = jnp.zeros_like(hand.cards)
    cards = hand.cards[hand.trump][:rank]
    return mask.at[hand.trump, jnp.where(cards > 0)].set(1)





def onAllTrump (trick, player, hand):
    abovetrumps = trumpsAbove(trick.best_card.rank, hand)
    return jax.lax.cond(
                jnp.any(abovetrumps), 
                lambda _: abovetrumps, # we can play above
                lambda _: jax.lax.cond(# we cannot play above
                    jnp.any(hand[trick.first_color]),
                    lambda _: hand[trick.first_color], #we have the color so we must play it
                    lambda _: hand #
                )
                operator=None

           )
def onSpecialBid (trick, player, hand):
    return jax.lax.cond(
                hand.trump == ALL_TRUMP,
                lambda _: onAllTrump (trick, player, hand),
                lambda _: onNoTrump (trick, player, hand),
                operator = None
           )
def allowedCards (trick, player, hand):
    current_color, current_rank = current_card.color, current_card.rank

    return jax.lax.cond(
                canCut(hand),
                lambda _: onNormalBid (trick, player, hand), #Standard color chosen
                lambda _: onSpecialBid (trick,player, hand), # NO_TRUMP or ALL_TRUMP
                operator = None
           )






