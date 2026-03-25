
from flax import struct
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as rnd

from Card import *
from Hand import *
from Trick import *
from LegalMoves import *


Player = Int [Array, "B"]

@struct.dataclass
class TrickState:
    """ Local environment describing how to play a trick """ 
    next_player : Player
    hands : Bool [Array, "B 4 4 8"]
    current_trick : Trick
    trick_value : Int [Array, "B"] # current value of the trick
    trick_size : Int [Array, "B"] # nb of cards played

def trickstate_to_tensor(ts : TrickState):
    return jnp.concatenate([
                ts.next_player,
                ts.hands.reshape(-1, 4*4*8),
                trick_to_tensor(ts.current_trick)],
                ts.trick_size,
                           axis=1)

def trickstate_initialize(first_players : Player, hands : Bool [Array, "B 4 4 8"]) -> TrickState:
    return TrickState(
                first_players,
                hands,
                new_trick(first_players),
                jnp.zeros(first_players.shape[0], dtype=int),
                jnp.zeros(first_players.shape[0], dtype=int))



# action = env.action_space()
def trickstate_actions (trump : Suit, trickstate : TrickState) -> Hand:
    player = trickstate.next_player
    #hands = trickstate.hands[player]
    hands = jax.vmap (lambda hand, p : hand[p])(trickstate.hands, player) 
    trick = trickstate.current_trick

    print (hands, hands.shape)

    return possible_moves(trump, trick, player, hands)


# n_obs, n_state, reward, done = env.step (key_step, state, action, env_params)

def trickstate_step (trump: Suit, trickstate : TrickState, card: Card) -> TrickState :

    def remove_card (player, card, hand):
        return hand[player].at[card.suit, card.rank].set(False)


    new_next_player = (trickstate.next_player + 1) % 4
    new_hand = jnp.vmap(remove_card)(trickstate.next_player, card, trickstate.hands)
    new_trick = play(trump, 
                     trickstate.current_trick,
                     trickstate.next_player,
                     card)
    new_trick_value = trickstate.trick_value + card_value(trump, card)
    new_trick_size = trickstate.trick_size + 1

    return TrickState(new_next_player, new_hand, new_trick, new_trick_value, new_trick_size)

    
# obs, state = env.reset(key)
