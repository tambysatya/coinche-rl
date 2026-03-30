
from flax import nnx

import jax.numpy as jnp



class MLP (nnx.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rngs, n_hid = 1):
        self.in_dim, self.hid_dim, self.out_dim = in_feats, hid_feats, out_feats
        self.n_hid = n_hid
        self.lin_in = nnx.Linear(in_feats, hid_feats, rngs = rngs)
        self.lin_out = nnx.Linear(hid_feats, out_feats, rngs = rngs)
        self.proj_in = nnx.Linear(in_feats, hid_feats, rngs=rngs)

        self.hid = []
        for i in range(n_hid-1):
            self.hid.append(nnx.Linear(hid_feats, hid_feats, rngs=rngs))

    @nnx.jit
    def __call__(self, x):
        y = self.lin_in(x)
        y = nnx.relu(y)+self.proj_in(x)
        for hid in self.hid:
            res = hid(y)
            res = nnx.relu(res)
            y = y+res
        y = self.lin_out(y)

        return y



class UniformPolicy (nnx.Module):
    def __init__(self):
        return
    def __call__(self, obs):
        batch_size = obs.shape[0]
        return jnp.ones([1,32])


