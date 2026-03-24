
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
class TrickState:
    
    next_player : Player
    hands :: Bool [Array, "B 4 4 8"],
    current_trick : Trick,

def trickstate_to_tensor(ts : TrickState):
    return jnp.concatenate([
                next_player,
                hands.reshape(-1, 4*4*8),
                trick_to_tensor(ts.current_trick)],
                           axis=1)
