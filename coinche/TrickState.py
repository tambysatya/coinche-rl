
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
from functools import partial

""" This modules describes an environment for simulating one complete trick (i.e. the 4 players have selected a card). """

Player = Int [Array, "B"]
@struct.dataclass
class TrickState:
    """ Local environment describing how to play a trick """ 
    next_player : Player
    hands : Bool [Array, "B 4 4 8"]
    current_trick : Trick
    trick_value : Int [Array, "B"] # current value of the trick
    trick_size : Int [Array, "B"] # nb of cards played

@jax.jit
def trickstate_to_tensor(ts : TrickState):
    """ Encoding to tensor: the next player and the trick_size are one-hot encoded """
    return jnp.concatenate([
                jax.nn.one_hot(ts.next_player,4),
                ts.hands.reshape(-1, 4*4*8),
                trick_to_tensor(ts.current_trick),
                ts.trick_value.reshape(-1,1),
                jax.nn.one_hot(ts.trick_size, 4)],
                           axis=1)

@jax.jit
def trickstate_obs_tensor (ts : TrickState):
    """ Returns the partial observation from a player (with only his hand)"""

    def extract_hand (player, hands):
        return hands[player]

    return jnp.concatenate ([
                jax.nn.one_hot(ts.next_player,4),
                jax.vmap(extract_hand)(ts.next_player, ts.hands).reshape(-1, 4*8),
                trick_to_tensor(ts.current_trick),
                ts.trick_value.reshape(-1,1),
                jax.nn.one_hot(ts.trick_size, 4)],
                             axis=1)
    
   
@jax.jit
def trickstate_initialize(first_players : Player, hands : Bool [Array, "B 4 4 8"]) -> TrickState:
    return TrickState(
                first_players,
                hands,
                new_trick(first_players),
                jnp.zeros(first_players.shape[0], dtype=int),
                jnp.zeros(first_players.shape[0], dtype=int))



# action = env.action_space()
@jax.jit
def trickstate_actions (trump : Suit, trickstate : TrickState) -> Hand:
    player = trickstate.next_player
    #hands = trickstate.hands[player]
    hands = jax.vmap (lambda hand, p : hand[p])(trickstate.hands, player) 
    trick = trickstate.current_trick


    return possible_moves(trump, trick, player, hands)

# n_obs, n_state, reward, done = env.step (key_step, state, action, env_params)
@partial(jax.jit, static_argnames=["enable_checks"])
def trickstate_step (trump: Suit, trickstate : TrickState, card: Card, enable_checks=False) -> TrickState :

    def remove_card (player, card, hand):
        return hand.at[player, card.suit, card.rank].set(False)


    new_next_player = (trickstate.next_player + 1) % 4
    new_hand = jax.vmap(remove_card)(trickstate.next_player, card, trickstate.hands)
    new_trick = play(trump, 
                     trickstate.current_trick,
                     trickstate.next_player,
                     card)
    new_trick_value = trickstate.trick_value + card_value(trump, card)
    new_trick_size = trickstate.trick_size + 1

    if enable_checks:
        err, _ = player_has_card(card, trickstate)
        err.throw()

        err, _ = is_move_allowed(trump, card, trickstate)
        err.throw()

    return TrickState(new_next_player, new_hand, new_trick, new_trick_value, new_trick_size)

@jax.jit
def trickstate_done(trickstate : TrickState) -> Bool [Array, "B"]:
    return trickstate.trick_size == 4
    
# obs, state = env.reset(key)


# Checkers 

@jax.jit
def player_has_card (card : Card, trickstate : TrickState):
    def scalar_player_has_card(player, card, hand):
        ret = hand[player]* card_to_subhand(card)
        checkify.check(jnp.any(ret), f"Invalid move: the player does not have this card (not in hand)")
        return ret
    @checkify.checkify
    def check ():
        ret = jax.vmap(scalar_player_has_card)(trickstate.next_player, card, trickstate.hands)
        return ret
    return check()

@jax.jit
@checkify.checkify
def is_move_allowed(trump: Suit, card : Card, trickstate : TrickState):
    allowed_moves = trickstate_actions(trump, trickstate)
    card = card_to_subhand(card)
    ret = jax.vmap(lambda c, allowed_move: jnp.any(c*allowed_move))(card, allowed_moves)
    checkify.check(jnp.all(ret), "Invalid move: the player is not allowed to play this card (not a legal move)")
    return ret


    


