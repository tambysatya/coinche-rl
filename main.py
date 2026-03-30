
from coinche.Card import *
from coinche.Hand import *
from coinche.Trick import *
from coinche.LegalMoves import *

from agents.rollout import *
from agents.basic_agent import *

import jax.random as rnd
import flax.nnx as nnx

from time import time
from tqdm import tqdm

seed = rnd.key(0)
policy_mdl = BasicAgent(10, nnx.Rngs(0))


def test(batch_size=1, seed=seed):
    step = mk_step(policy_mdl)
    params = nnx.state(policy_mdl)
    seed, key = rnd.split(seed)
    trick = new_trick(jnp.zeros(batch_size, dtype=int), deal(rnd.split(key, batch_size)))
    return step(params, None, jnp.zeros(batch_size, dtype=int),trick, key)


def statistics (batch_size, n_epoch, seed=seed):
    total_sum = 0
    for _ in tqdm(range(n_epoch)):
        seed, key = rnd.split(seed)
        start = time()
        test(batch_size=batch_size)
        stop = time()
        total_sum += stop-start

    print(f"{n_epoch}*{batch_size}: total={total_sum} avg={total_sum/n_epoch}")

