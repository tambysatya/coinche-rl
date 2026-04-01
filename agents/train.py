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
        pred = critic(trump, step.obs)
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



    def train_critic(params, 
                     trumps,step, reward,
                     n_epoch,batch_size=32,
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


def mk_train_actor (actor_mdl, critic_mdl):
    actor_graphdef, _ = nnx.split(actor_mdl)
    critic_graphdef, _ = nnx.split(critic_mdl)

    @jax.jit
    def compute_advantages(critic_params, trump, obs, reward):
        critic = nnx.merge(critic_graphdef, critic_params)
        return critic(trump, obs) - reward

    @jax.jit
    def ppo_loss (actor_current_params,
                  trump, records, advantages,
                  eps=0.2):
        new_policy = nnx.merge(actor_graphdef, actor_current_params)
        logprobs_new = jnp.log(new_policy(trump, records.obs)[0])
        ratio = jnp.exp(logprobs_new - records.logprobs)
        
        unclipped = ratio*advantages
        clipped = jnp.clip(ratio, 1-eps, 1+eps)*advantages
        loss = -jnp.mean(jnp.minimum(unclipped, clipped))
        return loss


    @partial(jax.jit, static_argnames=['optimizer'])
    def batch_scan (optimizer, #partially applied
                    batched_dataset, #partially applied
                    eps, #partially applied
                    carry,
                    batch_index):
        trump, records, advantages = batched_dataset
        params, opt_state = carry

        trump, advantages = trump[batch_index], advantages[batch_index]
        records = jtu.tree_map(lambda l: l[batch_index], records)

        loss, grads = jax.value_and_grad(ppo_loss)(params, trump, records, advantages, eps=eps)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return (params, opt_state), loss.mean()


    @partial(jax.jit, static_argnames=['optimizer'])
    def epoch_train (optimizer, batched_dataset, params, opt_state, eps=0.2):
        trumps, _, _ = batched_dataset
        batch_size = trumps.shape[0]
        return jax.lax.scan(partial(batch_scan, optimizer, batched_dataset, eps),
                            (params, opt_state),
                            jnp.arange(batch_size))


    def train_actor (critic_current_params, actor_current_params,
                     trump, records, rewards,
                     n_epoch, batch_size=32,
                     lr=0.1, eps=0.2):
        trump, rewards = mk_minibatches(trump, batch_size), mk_minibatches(rewards, batch_size)
        records = jtu.tree_map(lambda l: mk_minibatches(l, batch_size), records)
        

        print (f"Advantages evaluation")
        advantages = jax.vmap(partial(compute_advantages, critic_current_params))(trump, records.obs, rewards)

        optimizer = optax.adam(lr)
        opt_state = optimizer.init(actor_current_params)
        params = actor_current_params

        print (f"Actor optimization:")
        for i in range(n_epoch):
            (params, opt_state), value = epoch_train(optimizer,
                                                      (trump, records, advantages),
                                                      params, opt_state,
                                                      eps)
            print (f"[epoch={i}] loss: {value}")


    return train_actor
        





