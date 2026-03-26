
from agents.trick_policy import *
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


@partial(jax.jit, static_argnames= ['rollout_episode_fun', 'batch_size'])
def generate_examples(rollout_episode_fun, params, key, batch_size=2):
    """ Generates batch_size examples [state, value] """
    key, subkey = rnd.split(key)
    initial_player, trumps = jnp.zeros(batch_size, dtype=int), jnp.zeros(batch_size, dtype=int) #without loss of generality here
    initial_hands = deal(rnd.split(subkey, batch_size))
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

    

def training_loop(initial_params, dataset, lr=0.1, n_epoch=10):
    params = initial_params
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    obs, reward = dataset

    initial_loss = value_function_MC_loss(params, obs, reward)

    @jax.jit
    def step(params, opt_state):
        value, grads = jax.value_and_grad(value_function_MC_loss)(params, obs, reward)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value



    
    for _ in tqdm(range(n_epoch)):
        params, opt_state, value = step(params, opt_state)

    return params, initial_loss, value


def test_step(key=seed, batch_size=32, value_params=None):
    rollout_episode = mk_rollout_episode(policy_mdl)
    policy_params = nnx.state(policy_mdl)
    if value_params == None:
        value_params = nnx.state(value_mdl)
    dataset = generate_examples(rollout_episode, policy_params, key, batch_size=batch_size)
    final_params, initial_loss, value = training_loop(value_params, dataset)
    print (f"initial={initial_loss} final={value}")
    return final_params

def test_loop (key=seed, batch_size=32, nb_epoch=100):
    key, subkey = rnd.split(key)
    params = test_step(key=key, batch_size=batch_size)
    for i in range(nb_epoch):
        params = test_step(key=key, batch_size=batch_size, value_params=params)
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
