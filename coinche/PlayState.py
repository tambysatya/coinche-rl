
from flax import struct
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as rnd
import jax.tree_util as jtu

from coinche.Card import *
from coinche.Hand import *
from coinche.Trick import *
from coinche.TrickState import *
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


def playstate_initialize (bid : Bid, starting_players : Player, hands : Bool [Array, "B 4 4 8"]) -> PlayState:
    batch_size = starting_players.shape[0]
#    def generate_8_trickstates(hand, player):
#        hand = jnp.tile(hand, (8,1,1,1,1))
#        player = jnp.tile(player, (8,1)).swapaxes(1,0)
#        ts = trickstate_initialize(player, hand)
#        ts = jtu.tree_map(lambda leave: leave.squeeze(), ts)
#        return ts
    def generate_8_trickstates (hand, player):
        hand = jnp.expand_dims(hand,0).repeat(8,axis=0)
        player = jnp.repeat(player,8)
        current_tricks = new_trick(player)
        trick_value = jnp.zeros_like(player, dtype=int)
        trick_size = jnp.zeros_like(player, dtype=int)
        return TrickState (player, hand,current_tricks, trick_value, trick_size)
        
        
    return PlayState(bid,
                     jax.vmap(generate_8_trickstates)(hands, starting_players),
                     jnp.zeros(batch_size, dtype=int),
                     jnp.zeros([batch_size,2], dtype=int))
def playstate_is_done(playstate : PlayState):
    return playstate.current_trickstate_index == 8

def  playstate_next_trickstate (playstate : PlayState, finished_trick : TrickState):
    """ When the current trick is done, initializes a new trick """
    #def extract_finished_trick(trick, idx):
    #    return jtu.tree_map(lambda leaf: leaf[idx], trick)
    tricks = playstate.all_trickstates
    #finished_trick = jax.vmap(extract_finished_trick)(tricks, playstate.current_trickstate_index)
    final_hands = finished_trick.hands
    new_index = playstate.current_trickstate_index+1
    # computes who won the trick and initializes the next trickstate accordingly
    winners = finished_trick.current_trick.best_player

   # def set_new_trickstates (trickstates, index, player, hands):
   #     new_trickstate = trickstate_initialize(player,hands)
   #     return jtu.tree_map(lambda f,g: f.at[index].set(g), trickstates, new_trickstate)
   #         
   # new_trickstates = jax.vmap(set_new_trickstates)(tricks,new_index, winners, final_hands)

    def set_new_trickstate (all_8_trickstates, index, new_trickstate):
        return jtu.tree_map(lambda f, g: f.at[index].set(g), all_8_trickstates, new_trickstate)
    new_trickstates = TrickState(winners, final_hands, new_trick(winners), jnp.zeros_like(winners, dtype=int), jnp.zeros_like(winners, dtype=int))

    new_trickstates = jax.vmap(set_new_trickstate)(tricks, new_index, new_trickstates)

    trick_score = finished_trick.trick_value[:,None]
    trick_score_win = jnp.concatenate([trick_score, jnp.zeros_like(trick_score, dtype=int)], axis=1) # team [0,2] wins
    trick_score_lose = jnp.concatenate([jnp.zeros_like(trick_score, dtype=int), trick_score], axis=1) # team [1,3] wins
    cond_win = winners%2 == 0 
    trick_score = cond_win*trick_score_win + (1-cond_win)*trick_score_lose

    return PlayState(playstate.bid,
                     new_trickstates,
                     new_index,
                     playstate.current_scores + trick_score)


