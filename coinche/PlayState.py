
from flax import struct
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as rnd
from jax.experimental import checkify

from coinche.Card import *
from coinche.Hand import *
from coinche.Trick import *
from coinche.LegalMoves import *
from coinche.Bid import *
from functools import partial

""" This environment simulates the Phase II of the game: given a bid (selected in phase 1) the players have to play 8 tricks (see TrickState.py)"""

@struct.dataclass
class PlayState:
    bid : Bid # B Bid
    all_trickstates : TrickState # [B 8 TrickState]
    current_trickstate_index : Int [Array, "B"] #index of the trickstate that is currently played
    current_scores : Int [Array, "B 2"] #current score of each team


def playstate_initialize (bid : Bid, starting_player : Player, hands : Bool [Array, "B 4 4 8"]) -> PlayState:
    batch_size = starting_player.shape[0]
    def generate_8_trickstates(_):
        return jax.vmap(lambda _: trickstate_initialize(starting_player, hands))(jnp.arange(8))

    return PlayState(bid,
                     jax.vmap(generate_8_trickstates)(jnp.arange(batch_size))
                     jnp.zeros(batch_size, dtype=int),
                     jnp.zeros(batch_size, dtype=int))
def playstate_is_done(playstate : PlayState):
    return playstate.current_trickstate_index == 8

def  playstate_next_trickstate (playstate : PlayState):
    """ When the current trick is done, initializes a new trick """
    tricks = playstate.all_trickstates
    finished_trick = jax.vmap(lambda trick, idx: trick[idx])(tricks, playstate.current_trickstate_index)
    final_hands = finished_trick.hands
    new_index = playstate.current_trickstate_index+1
    # computes who won the trick and initializes the next trickstate accordingly
    winners = finished_trick.current_trick.best_player

    def set_new_trickstates (trickstates, index, player):
        return trickstates.at[index].set(
                    trickstate_initialize(player,final_hands)
                )
    
    new_trickstates = jax.vmap(set_new_trickstates)(tricks,new_index, winners)
    trick_score = jnp.concatenate([(winners % 2)*finished_trick.trick_value,
                                   (1-(winners%2))*(-finished_trick.trick_value)])

    return PlayState(playstate.bid,
                     new_trickstates,
                     new_index,
                     playstate.current_scores + trick_score)


