import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as rnd
import jax.tree_util as jtu
import flax.nnx as nnx

from functools import partial

from utils import *
from coinche.Card import *
from agents.rollout.game import *


@jax.jit
def compute_transition_rewards (dicount_factor : Int,
                                bid : BidRecord,
                                trick_hist : TrickHistory, # B : final history of each game (trick_hist.index = 8)
                                tricks : Trick, # 8 B
                                bidding_steps : BidStep, # 10,4,B
                                traj_steps : TrickStep): # 8,4,B

    # Attacking with a specific bid applies an initial malus (you starts the game with a negative score equal to the rank of the bid)

    attacker_team = bid.author % 2 # shape B
    attacker_score = jax.vmap(lambda score, team : score[team])(trick_hist.team_scores, attacker_team)

    contract_value = bid.bid.rank.argmax(axis=-1)*10 + 80 # shape B, between 80 and 160 (last is ALL_IN)
    has_someone_bet_p = bid.bid.rank.any(axis=-1) == True # shape B
    has_bet_all_in_p = bid.bid.rank[:,-1] # shape B
    has_made_all_in_p =  ((trick_hist.tricks.best_player % 2) == attacker_team[:,None]).all(axis=-1) # shape B

    attackers_won_p = has_bet_all_in_p*has_made_all_in_p + (1-has_bet_all_in_p)*(attacker_score >= contract_value)
    # adds 250 to the attacker_score if all_in, else adds the contract value
    attackers_won_score = jax.vmap(
                            lambda team_scores, a_team, val, all_in_p: 
                                   team_scores.at[a_team].add(all_in_p*250 + (1-all_in_p)*val))(trick_hist.team_scores, attacker_team, contract_value, has_bet_all_in_p)
    attackers_lose_score = jax.vmap(lambda team_scores, a_team: team_scores.at[a_team].set(0).at[~a_team].add(160))(trick_hist.team_scores, attacker_team)

    return jnp.where(attackers_won_p[:,None], attackers_won_score, attackers_lose_score)






