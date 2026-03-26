

from nn.blocks import *
from coinche.TrickState import *
from coinche.Card import *
from jaxtyping import Array, Int, Bool

import jax.numpy as jnp
import jax
import jax.random as rnd

key = rnd.key(0)


def mk_policy_step (model):
    """ Generates policy_step() optimized for a given model (policy network) architecture """
    graphdef, _ = nnx.split(model)

    def policy_step (trump : Suit, trickstate : TrickState, params, keys) -> Card : 
        """ Sample an action from a policy """
        policy_network = nnx.merge(graphdef, params)
        logits = policy_network(trickstate_obs_tensor(trickstate))
        mask = trickstate_actions(trump, trickstate)
        probs = jnp.where(mask, logits, -jnp.inf)
        action = jax.vmap(jax.random.categorical)(keys, probs.reshape(-1,32))

        suit, rank = action // 8, action % 8
        return Card(suit, rank)
    return jax.jit(policy_step)

def mk_rollout_episode(model):
    """ Generates rollout_episode() optimized for a given model (policy network) architecture """

    policy_step = mk_policy_step(model)

    def rollout_episode (trump : Suit, initial_state : TrickState, params, key):
        """ Batched rollout on multiple states.
            Returns a batch of trajectories, i.e. a batched trickstate of size [4, B]
        """
        def scan_step (state, carry_key):
            subkey = rnd.split(carry_key, batch_size)
            card = policy_step(trump, state, params, subkey)
            new_state = trickstate_step(trump, state, card)
            return new_state, new_state

        batch_size = trump.shape[0]
        initial_keys = rnd.split(key, 4)
        final_states, trajectories = jax.lax.scan(scan_step, initial_state, initial_keys, length=4)
        return final_states, trajectories

    return jax.jit(rollout_episode)
            




