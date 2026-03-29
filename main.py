
from agents.trick_policy import *
from agents.play_policy import *
from nn.blocks import *
import jax.random as rnd

import jax.tree_util as jtu
from time import time
from tqdm import tqdm
from functools import partial
import optax


seed = rnd.key(0)
seed, mdlseed = rnd.split(seed)
policy_mdl = MLP(in_feats=110, hid_feats=100, out_feats=4*8, rngs=nnx.Rngs(mdlseed), n_hid=1)
value_mdl = MLP(in_feats=110, hid_feats=10, out_feats=1, rngs=nnx.Rngs(mdlseed), n_hid=1)
value_graphdef, _ = nnx.split(value_mdl)
policy_graphdef, _ = nnx.split(policy_mdl)
#mdl = UniformPolicy()


@partial(jax.jit, static_argnames= ['rollout_episode_fun', 'n_examples'])
def generate_examples(rollout_episode_fun, params, key, n_examples=2):
    """ Generates n_examples examples [state, value] """
    key, subkey = rnd.split(key)
    initial_player, trumps = jnp.zeros(n_examples, dtype=int), jnp.zeros(n_examples, dtype=int) #without loss of generality here
    initial_hands = deal(rnd.split(subkey, n_examples))
    initial_state = trickstate_initialize(initial_player, initial_hands) 


    final_states, trajectories = rollout_episode_fun(trumps, initial_state, params, key)
    
    # trajectories is a tensor [T, B, Trick] where T = 4 (the horizon) and B is the number of tricks (batch size)
    # we want a tensor [T*B, Trick]: we need to reshape the leafs of the pytree 

    trajectories = jtu.tree_map (lambda leaf: leaf.reshape(leaf.shape[0]*leaf.shape[1], *leaf.shape[2:]), trajectories)

    #Identifies the reward = +/- trick value  depending on whether the team won or not
    final_states = jtu.tree_map (lambda leaf: leaf.reshape(-1,1).repeat(4, axis=1), final_states) #generates one final step per intermediate step: we transform a [B, FinalState] into a [B,4,FinalState]
    final_states = jtu.tree_map (lambda leaf: leaf.reshape(leaf.shape[0]*leaf.shape[1], *leaf.shape[2:]), final_states) #As for the trajectories, we transform a [4, B, FinalState] into a  [4*B, FinalState]

    #reward = jax.vmap(compute_reward) (trajectories, final_states)
    reward = jnp.where((trajectories.next_player % 2) == (final_states.current_trick.best_player % 2),
                       final_states.trick_value,
                       - final_states.trick_value)
    obs = trickstate_obs_tensor(trajectories)
    return obs, reward
    
@jax.jit
def value_function_MC_loss(params, obs, reward):
    model = nnx.merge(value_graphdef, params)
    pred = jax.vmap(model)(obs)
    return ((pred - reward)**2).mean()

@partial(jax.jit, static_argnames=['optimizer'])
def step(optimizer, batched_dataset, carry, i):
    params, opt_state = carry
    obs, reward = batched_dataset[0][i], batched_dataset[1][i]

    value, grads = jax.value_and_grad(value_function_MC_loss)(params, obs, reward)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return (params, opt_state), value


@partial(jax.jit, static_argnames=['optimizer'])
def train_epoch(optimizer, batched_dataset, params, opt_state):
    return jax.lax.scan( partial(step, optimizer, batched_dataset),
                         (params, opt_state),
                         jnp.arange(batched_dataset[0].shape[0])
                        )




def training_loop(initial_params, dataset, batch_size, lr=0.1, n_epoch=10):
    params = initial_params
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)



    batched_obs, batched_reward = mk_minibatches(dataset[0], batch_size), mk_minibatches(dataset[1], batch_size)



    
    for i in tqdm(range(n_epoch)):
        (params, opt_state), value = train_epoch(optimizer, (batched_obs, batched_reward), params, opt_state)
        print (f"\t [epoch {i}] Loss={value.mean()}")

    return params, value.mean()


def test_step(key=seed, n_examples=32, batch_size=32, value_params=None):
    rollout_episode = mk_rollout_episode(policy_mdl)
    policy_params = nnx.state(policy_mdl)
    if value_params == None:
        value_params = nnx.state(value_mdl)
    dataset = generate_examples(rollout_episode, policy_params, key, n_examples=n_examples)
    final_params, value = training_loop(value_params, dataset, batch_size)
    print (f"final={value}")
    return final_params

def test_loop (key=seed, n_examples=32, batch_size=32, nb_epoch=100):
    key, subkey = rnd.split(key)
    params = nnx.state(value_mdl)
    for i in range(nb_epoch):
        params = test_step(key=key, n_examples=n_examples, batch_size=batch_size, value_params=params)
    return params



def compute_stats(n=100, batch_size=2):
    rollout_episode = mk_rollout_episode(mdl)
    initial_params = nnx.state(mdl)
    key = seed

    stat = 0
    print (f"Generating {n} batchs of size {batch_size}")
    for i in tqdm(range(n)):
        key, _ = rnd.split(key)
        start = time()
        ret = generate_examples(rollout_episode, initial_params,key, batch_size=batch_size)
        stat += time() - start
    print (f"Average {stat/n}s per trajectory")
    return stat/n
