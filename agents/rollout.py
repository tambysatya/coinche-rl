from flax import struct
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as rnd
import jax.tree_util as jtu
import flax.nnx as nnx

from coinche.Trick import *
from coinche.LegalMoves import *

@struct.dataclass
class Step:
    # Inputs of the policy network
    obs : jax.Array # observation of the state 
    hidden_state : jax.Array # embedding of the past experiences (to augment the state with an embedding, produced by the previous calls of the policy network, and carried through the trajectory).
    #Action 
    logprobs : jax.Array # log-probability inferred by the network


def mk_step(policy_model):
    graphdef, _ = nnx.split(policy_model)

    def step (params, hidden_state,
              trump : Suit, trick : Trick,
              key) -> Trick :
        obs = trick_obs(trick)
        policy = nnx.merge(graphdef, params)

        legal_moves = possible_moves(trump, trick)
        logits, next_hidden_state = policy(trump, obs, hidden_state)
        probas = jnp.where(legal_moves.reshape([-1,32]),
                           logits,
                           -jnp.inf)
        probas = jax.nn.softmax(probas, axis=1)
        action = rnd.categorical(key, probas)
        card = card_from_index(action)

        record = Step(obs, hidden_state, jnp.log(probas))

        return record, play(trump, trick, card)
    
    return jax.jit(step)

        

        


