
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
    bid : Bid
    all_trickstates : TrickState # [B 8 TrickState]
    current_trickstate_index : Int [Array, "B"] #index of the trickstate that is currently played
    current_scores : Int [Array, "B 2"] #current score of each team


#def playstate_initialize (bid : Bid, starting_player : Player, hands : Bool [Array, "B 4 4 8"]) -> PlayState:
def playstate_is_done(playstate : PlayState):
    return playstate.current_trickstate_index == 8
def  playstate_next_trickstate (playstate : PlayState):
    tricks = playstate.all_trickstates
    finished_trick = jax.vmap(lambda trick, idx: trick[idx])(tricks, playstate.current_trickstate_index)
    # computes who won the trick and initializes the next trickstate accordingly

    def set_new_trickstates (trickstates, index, player):
        return trickstates.at[index].set(
                    trickstate_initialize(player, )
                )
    
    new_trickstates = jax.vmap(set_new_trickstates)

    return PlayState(playstate.bid,
                     new_trickstates,
                     playstate.current_trickstate_index+1,
                     playstate.current_scores)
