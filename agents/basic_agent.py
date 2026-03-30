
import jax.numpy as jnp
from nn.blocks import *


class BasicAgent (nnx.Module):
    def __init__(self, hid_feats, rngs, n_hid = 1):
        in_feats = 1 + 97 # Trump + obs_tensor. Nno hidden state yet
        out_feats = 32
        self.mlp = MLP(in_feats, hid_feats, out_feats, rngs, n_hid=n_hid)

    @nnx.jit
    def __call__ (self, trump, obs, hidden_state): 
        x = jnp.concat([trump[:,None], obs], axis=1)
        return self.mlp(x), jnp.zeros([x.shape[0],1])
