
from flax import struct
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as rnd

from Card import *
from Hand import *
from Trick import *


Player = Int [Array, "B"]

@struct.dataclass
class GameState:
    hands : Bool [Array, "B 4 4 8"] # The Hands of the 4 players
    current_tricks : Trick # One trick per state
    current_players : Player # Who plays the next move

    trick_has_started_p : Bool [Array, "B"] # False: the trick is empty (=> legalmoves should return the hand)




   
    

@jax.jit
def reset(key, starting_players : Player) -> GameState :
    batch_size = starting_players.shape[0]
    subkeys = rnd.split(key, batch_size)
    hands = deal(subkeys)

    trick_has_started_p = jnp.zeros(batch_size, dtype=bool)
    dummy_tricks = new_trick(starting_players,
                             Card(jnp.zeros(batch_size, dtype=int),
                             jnp.zeros(batch_size, dtype=int)))

    return GameState(hands, dummy_tricks, starting_players, trick_has_started_p)

