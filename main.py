
from coinche.Card import *
from coinche.Hand import *
from coinche.Trick import *
from coinche.LegalMoves import *

from agents.rollout import *
from agents.basic_agent import *

import jax.random as rnd
import flax.nnx as nnx

from functools import partial

from time import time
from tqdm import tqdm

seed = rnd.key(0)
policy_mdl = BasicAgent(10, nnx.Rngs(0))


@partial(jax.jit, static_argnames=["batch_size"])
def test(batch_size=1, seed=seed):
    #rollout = mk_rollout(policy_mdl)
    #rollout = mk_trick_rollout(policy_mdl)
    step = mk_step(policy_mdl)
    params = nnx.state(policy_mdl)
    seed, key = rnd.split(seed)
    trumps = jnp.zeros(batch_size, dtype=int)
    trick = new_trick(jnp.zeros(batch_size, dtype=int), deal(rnd.split(key, batch_size)))
    dummy_step = Step (trick_obs(trick), jnp.zeros([batch_size,1]), jnp.zeros([batch_size, 32]))
    initial_players = jnp.zeros(batch_size, dtype=int)
    initial_hands = deal(rnd.split(key, batch_size))
    #return rollout(params, dummy_step.hidden_state, trumps, initial_players, initial_hands, seed)
    #return rollout(params, dummy_step, trumps, initial_players,  initial_hands, seed)
    return trick, step(params, None, jnp.zeros(batch_size, dtype=int),trick, key)


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
    rollout = mk_trick_rollout(policy_mdl)
    params = nnx.state(policy_mdl)
    seed, key = rnd.split(seed)
    trumps = jnp.zeros(batch_size, dtype=int)
    trick = new_trick(jnp.zeros(batch_size, dtype=int), deal(rnd.split(key, batch_size)))
    step = Step (trick_obs(trick), jnp.zeros([batch_size,1]), jnp.zeros([batch_size, 32]))
    players = jnp.zeros(batch_size, dtype=int)
    hands = deal(rnd.split(key, batch_size))
    total = 0
    for i in range(8):
        (trick, step), traj = rollout(params, step, trumps, players, hands, seed)
        players = trick.best_player
        hands = trick.hands
        total += trick.value
        print (f"{trick.value} (total={total}) players={players}")



