
import jax.numpy as jnp
from nn.blocks import *
from coinche.Bid import *


class BasicAgent (nnx.Module):
    def __init__(self, hid_feats, rngs, n_hid = 1):
        in_feats = 1 + 97 + 2 # Trump + obs_tensor + current/total score. Nno hidden state yet
        out_feats = 32
        self.mlp = MLP(in_feats, hid_feats, out_feats, rngs, n_hid=n_hid)

    @nnx.jit
    def __call__ (self, trump, obs): 
        x = jnp.concat([trump[:,None], obs.trick, obs.current_score[:,None], obs.total_score[:,None]], axis=1)
        return self.mlp(x), jnp.zeros([x.shape[0],1])


class BasicCritic (nnx.Module):
    def __init__(self, hid_feats, rngs, n_hid = 1):
        in_feats = 1 + 97 + 2 # Trump + obs_tensor + current/total score. Nno hidden state yet
        self.mlp = MLP(in_feats, hid_feats, 1, rngs, n_hid=n_hid)

    @nnx.jit
    def __call__ (self, trump, obs): 
        x = jnp.concat([trump[:,None], obs.trick, obs.current_score[:,None], obs.total_score[:,None]], axis=1)
        return self.mlp(x)



class BidAgent (nnx.Module):
    def __init__(self, hid_feats, rngs, n_hid = 1):
        in_feats = 54 # obs tensor
        out_feats = 16
        self.mlp = MLP(in_feats, hid_feats, out_feats, rngs, n_hid=n_hid)

    @nnx.jit
    def __call__ (self, obs): 
        x = jnp.concat([obs.hidden_state,
                        bid_to_tensor(obs.bid), 
                        obs.author[:,None],
                        obs.checked,
                        obs.hand.reshape([-1,32])],
                        axis=1)
        x = self.mlp(x)
        logit_pass, logit_suit, logit_rank = x[:,:2],x[:, 2:7],x[:, 7:16]
        return (logit_pass, logit_suit, logit_rank), obs.hidden_state



