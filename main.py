
from coinche.Card import *
from coinche.Hand import *
from coinche.Trick import *
from coinche.LegalMoves import *

from agents.rollout import *
from agents.basic_agent import *
from agents.train import *

import jax.random as rnd
import flax.nnx as nnx

from functools import partial

from time import time
from tqdm import tqdm

seed = rnd.key(0)
policy_mdl = BasicAgent(10, nnx.Rngs(0))
critic_mdl = BasicCritic(10, nnx.Rngs(0))


@partial(jax.jit, static_argnames=["batch_size"])
def test(batch_size=1, seed=seed):
    rollout_full = mk_rollout(policy_mdl)
    #rollout = mk_trick_rollout(policy_mdl)
    #step = mk_step(policy_mdl)
    params = nnx.state(policy_mdl)
    seed, key = rnd.split(seed)
    trumps = jnp.zeros(batch_size, dtype=int)
    trick = new_trick(jnp.zeros(batch_size, dtype=int), deal(rnd.split(key, batch_size)))
    #dummy_step = Step (trick_obs(trick), jnp.zeros([batch_size,1]), jnp.zeros([batch_size]))
    initial_hidden_state = jnp.zeros([batch_size,1])
    initial_players = jnp.zeros(batch_size, dtype=int)
    initial_hands = deal(rnd.split(key, batch_size))
    initial_scores = jnp.zeros([batch_size,2])
    total_score = jnp.zeros([batch_size])
    return rollout_full(params, initial_hidden_state, trumps, initial_players, initial_hands, seed)
    #return rollout(params, initial_hidden_state, trumps, initial_scores, total_score, initial_players,  initial_hands, seed)
    #return trick, step(params, None, jnp.zeros(batch_size, dtype=int),trick, key)


def statistics (batch_size, n_epoch, seed=seed):
    total_sum = 0
    for _ in tqdm(range(n_epoch)):
        seed, key = rnd.split(seed)
        start = time()
        test(batch_size=batch_size)
        stop = time()
        total_sum += stop-start

    print(f"{n_epoch}*{batch_size}: total={total_sum} avg={total_sum/n_epoch}")

def dbg_scan(batch_size=1, seed=seed):
    step = mk_step(policy_mdl)
    rollout = mk_trick_rollout(policy_mdl)
    params = nnx.state(policy_mdl)
    seed, key = rnd.split(seed)
    trumps = jnp.zeros(batch_size, dtype=int)
    trick = new_trick(jnp.zeros(batch_size, dtype=int), deal(rnd.split(key, batch_size)))
    record = Step (trick_obs(trick), jnp.zeros([batch_size,1]), jnp.zeros([batch_size]))

    players = jnp.zeros(batch_size, dtype=int)
    hands = deal(rnd.split(key, batch_size))
    total = 0
    for i in range(8):
        #seed,key = rnd.split(seed)
        #(trick, record), traj = rollout(params, init_step, trumps, players, hands, key)
        (trick, record) = step(params, None, jnp.zeros(batch_size, dtype=int),trick, key)
        players = trick.best_player
        hands = trick.hands
        total += trick.value
        print (f"{trick.value} (total={total}) players={players} next_player={trick.current_player}  card_per_player={hands[0].sum(axis=(1,2))} trick={trick.cards}")
        #print (f"{trick.value} (total={total}) players={players} card_per_player={hands[0].sum(axis=(1,2))}, traj.shape={traj.obs.shape}")



def test_samples(batch_size=32, discount_factor=0.9, seed=seed):
    players = jnp.zeros(batch_size,dtype=int)
    trump = jnp.zeros(batch_size,dtype=int)
    hidden_states = jnp.zeros([batch_size,1])

    collect_samples = mk_collect_samples(policy_mdl)

    seed, key = rnd.split(seed)

    records, rewards = collect_samples(discount_factor, 
                                       nnx.state(policy_mdl),
                                       hidden_states, trump, players, deal(rnd.split(key,batch_size)), seed)

    return records, rewards



def test_ppo (n_trajectory_samples=32, batch_size=32, n_epoch=100, discount_factor=0.9,lr=0.05, seed=seed):
    
    train_actor, train_critic = mk_train_actor(policy_mdl, critic_mdl), mk_train_critic(critic_mdl)

    print ("Rollout...")
    records, rewards = test_samples(batch_size=n_trajectory_samples, discount_factor=discount_factor, seed=seed)

    trumps = jnp.zeros(n_trajectory_samples*32) # mult by 32 because we have 32 sample per game
    critic_params = train_critic(nnx.state(critic_mdl),
                                 trumps, records, rewards, 
                                 n_epoch, batch_size=batch_size, lr=lr)

    actor_params = train_actor(critic_params, #WARNING: use the new critic values
                               nnx.state(policy_mdl),
                               trumps, records, rewards,
                               n_epoch, batch_size=batch_size, lr=lr, eps=0.2)




