

from nn.blocks import *
from coinche.TrickState import *
from coinche.Card import *

import jax.numpy as jnp
import jax



def rollout (trump : Suit, trickstate : TrickState, policy_network, keys) : 
    logits = policy_network(trickstate_obs_tensor(trickstate))
    mask = trickstate_actions(trump, trickstate)
    probs = jnp.where(mask, logits, -jnp.inf)
    action = jax.random.categorical(keys, probs)
    



