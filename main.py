
from coinche.Card import *
from coinche.Hand import *
from coinche.Trick import *
from coinche.LegalMoves import *

from agents.rollout.trick import *
from agents.rollout.bid import *
from agents.rollout.game import *
from agents.basic_agent import *
from agents.train import *
from agents.rollout.reward import *

import jax.random as rnd
import flax.nnx as nnx

from functools import partial

from time import time
from tqdm import tqdm

seed = rnd.key(0)
game_policy_mdl = BasicAgent(10, nnx.Rngs(0))
critic_mdl = BasicCritic(10, nnx.Rngs(0))

bid_policy_mdl = BidAgent(10,nnx.Rngs(0)) #, uniform=True)


def generate_actor_pool (pool_size, model):
    pool = [nnx.state(model) for i in range (pool_size)]
    return jtu.tree_map(lambda *f: jnp.stack(f), *pool)

@partial(jax.jit, static_argnames=["pool_size", "game_per_pair"])
def test_game_rollout (pool_size = 2, game_per_pair=4, seed = seed):
    all_bid_params = generate_actor_pool(pool_size, model=bid_policy_mdl)
    all_game_params = generate_actor_pool(pool_size, model=game_policy_mdl)
    batch_size = game_per_pair*(pool_size ** 2)

    agent_indices = jnp.indices((pool_size, pool_size)).T
    agent_indices = jnp.tile(agent_indices,(1,1,game_per_pair)).reshape([batch_size,2])
    permutation = build_permutation(agent_indices)
    hidden_states = jnp.zeros([batch_size,4,3]) # 4 hidden states per game
    initial_player = jnp.zeros([batch_size], dtype=int)

    rollout = mk_rollout(bid_policy_mdl, game_policy_mdl, pool_size)
    print ("Let's go")
    ret = rollout(all_bid_params, all_game_params,
                  permutation, hidden_states,
                  initial_player,
                  seed)
    bid, final_trick_history, traj_tricks, bidding_steps, traj_steps = ret
    scores = compute_transition_rewards(0.99, bid, final_trick_history, traj_tricks, bidding_steps, traj_steps)
    return ret, scores



@partial(jax.jit, static_argnames=["batch_size", "pool_size"])
def test(batch_size=4, pool_size=2, seed=seed):
    rollout_full = mk_rollout(bid_policy_mdl, pool_size)
    #rollout = mk_trick_rollout(policy_mdl, pool_size)
    #step = mk_step(policy_mdl)
    #league_step = mk_league_step(policy_mdl, pool_size)
    params = nnx.state(policy_mdl)
    initial_params = generate_actor_pool(pool_size)
    agent_indices = jnp.zeros([batch_size], dtype=int)
    game_indices = jnp.concat([jnp.zeros([batch_size,1]), jnp.ones([batch_size,1])], axis=1)
    permutations = game_indices.argsort(axis=1)
    seed, key = rnd.split(seed)
    trumps = jnp.zeros(batch_size, dtype=int)
    trick = new_trick(jnp.zeros(batch_size, dtype=int), deal(rnd.split(key, batch_size)))
    #dummy_step = Step (trick_obs(trick), jnp.zeros([batch_size,1]), jnp.zeros([batch_size]))
    initial_hidden_state = jnp.zeros([batch_size,4,1])
    initial_players = jnp.zeros(batch_size, dtype=int)
    initial_hands = deal(rnd.split(key, batch_size))
    initial_scores = jnp.zeros([batch_size,2])
    total_score = jnp.zeros([batch_size])
    #return rollout_full(params, initial_hidden_state, trumps, initial_players, initial_hands, seed)
    #return league_step(initial_params, agent_indices, initial_hidden_state, trumps, trick, initial_scores[0], total_score, key)
    #return rollout(initial_params, game_indices, initial_hidden_state, trumps, initial_scores, total_score, initial_players, initial_hands, key)
    return rollout_full(initial_params, permutations, initial_hidden_state, trumps, initial_players, initial_hands, seed)


def statistics (game_per_pair, pool_size, n_epoch, seed=seed):
    total_sum = 0
    for _ in tqdm(range(n_epoch)):
        seed, key = rnd.split(seed)
        start = time()
        #test(batch_size=batch_size, pool_size=pool_size, seed=key)
        traj_tricks, bidding_steps, traj_steps = test_game_rollout(pool_size=pool_size, game_per_pair=game_per_pair, seed=key)
        stop = time()
        total_sum += stop-start

    print(f"{n_epoch}*{game_per_pair*pool_size**2}: total={total_sum} avg={total_sum/n_epoch}")

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



def test_samples(pool_size = 8, game_per_pair=5, discount_factor=0.9, seed=seed):

    batch_size = game_per_pair*(pool_size ** 2)
    agents_indices = jnp.indices((pool_size, pool_size)).T
    agents_indices = jnp.tile(agents_indices,(1,1,game_per_pair)).reshape([batch_size,2])

    all_params = generate_actor_pool(pool_size)


    players = jnp.zeros(batch_size,dtype=int)
    trump = jnp.zeros(batch_size,dtype=int)
    hidden_states = jnp.zeros([batch_size,1])

    collect_samples = mk_collect_samples(policy_mdl, pool_size)

    seed, key = rnd.split(seed)

    records, rewards = collect_samples(discount_factor, 
                                       all_params, agents_indices,
                                       hidden_states, trump, players, deal(rnd.split(key,batch_size)), seed)

    return records, rewards



def test_ppo (n_trajectory_samples=32, batch_size=32, n_epoch=100, discount_factor=0.9,lr=0.05, seed=seed, eps=0.2):
    
    train_actor, train_critic = mk_train_actor(policy_mdl, critic_mdl), mk_train_critic(critic_mdl)

    print ("Rollout...")
    records, rewards = test_samples(batch_size=n_trajectory_samples, discount_factor=discount_factor, seed=seed)

    assert jnp.all(jnp.isfinite(rewards)), "Inifinte rewards"

    trumps = jnp.zeros(n_trajectory_samples*32) # mult by 32 because we have 32 sample per game
    critic_params = train_critic(nnx.state(critic_mdl),
                                 trumps, records, rewards, 
                                 n_epoch, batch_size=batch_size, lr=lr)

    actor_params = train_actor(critic_params, #WARNING: use the new critic values
                               nnx.state(policy_mdl),
                               trumps, records, rewards,
                               n_epoch, batch_size=batch_size, lr=lr, eps=eps)




def test_bid(pool_size=2, game_per_pair=3, seed=seed):
    all_params = generate_actor_pool(pool_size, model=bid_policy_mdl)
    batch_size = game_per_pair*(pool_size ** 2)

    agent_indices = jnp.indices((pool_size, pool_size)).T
    agent_indices = jnp.tile(agent_indices,(1,1,game_per_pair)).reshape([batch_size,2])
    permutation = agent_indices.argsort(axis=1)

    seed, key = rnd.split(seed)
    hands = deal(rnd.split(key, batch_size))
    hidden_states = jnp.zeros([batch_size, 4, 5])
    checked = jnp.zeros([batch_size, 4], dtype=bool)

    current_player = jnp.zeros([batch_size], dtype=int)

    history = history_initialize(current_player)

    predict_bid = mk_predict_bid(bid_policy_mdl, pool_size)
    bid_scan = mk_bid_scan(bid_policy_mdl, pool_size)
    bid_rollout = mk_bid_rollout(bid_policy_mdl, pool_size)
    bidding_count = jnp.zeros(batch_size, dtype=int)
    carry = hidden_states, bidding_count, history, seed
    player_orders = jnp.tile(jnp.arange(4),(batch_size,1))

    return bid_rollout(all_params, permutation, hands, hidden_states, current_player, seed)
    #return bid_scan(all_params,player_orders, permutation, hands, carry, 0)
    #return history, predict_bid (nnx.state(bid_policy_mdl), seed, current_player, hands, hidden_states, bidding_count, history)
    #bid_scan = mk_bid_rollout(bid_policy_mdl, pool_size)
#    return bid_scan(all_params, permutation, hands,
#                    (hidden_states, initial_bid, checked, seed),
#                    current_player)
#


