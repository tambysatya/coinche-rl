from flax import struct
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as rnd
import jax.tree_util as jtu
import flax.nnx as nnx
import optax

from utils import *
from coinche.Trick import *
from coinche.LegalMoves import *
from agents.rollout import *



def mk_train_critic(critic_mdl):
    graphdef, _ = nnx.split(critic_mdl)
    
    @jax.jit
    def loss_function (params, 
                       trump, step, reward):
        critic = nnx.merge(graphdef, params)
        pred = critic(trump, step.obs, step.hidden_state)
        return ((pred - reward)**2).mean()


    @partial(jax.jit, static_argnames=['optimizer'])
    def epoch_train (optimizer,
                     initial_carry, # = initial_params, initial_opt_state,
                     batched_dataset):
        trump, step, reward = batched_dataset
        def batch_scan(carry, batch_index):
            params, opt_state = carry
            step_batch = jtu.tree_map(lambda l: l[batch_index], step)
            value, grads = jax.value_and_grad(loss_function)(params,trump[batch_index], step_batch, reward[batch_index])
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), value 
        final_carry, values = jax.lax.scan(batch_scan, initial_carry, jnp.arange(trump.shape[0]))
        return final_carry, values.mean()



    def train_critic(params, trumps,
                     step, reward,
                     n_epoch,
                     batch_size=32,
                     lr = 0.1):
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)
        
        trump = mk_minibatches(trumps, batch_size)
        step = jtu.tree_map(lambda l: mk_minibatches(l, batch_size), step)
        reward = mk_minibatches(reward, batch_size)
        batched_dataset = trump, step, reward

        print (f"Critic optimization:")
        for i in range(n_epoch):
            (params, opt_state), value = epoch_train(optimizer, (params, opt_state), batched_dataset)
            print (f"[epoch={i}] loss: {value}")

        return params


    return train_critic




