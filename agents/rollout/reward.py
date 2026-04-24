import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as rnd
import jax.tree_util as jtu
import flax.nnx as nnx
from jaxtyping import Float

from functools import partial

from utils import *
from coinche.Card import *
from agents.rollout.game import *


@jax.jit
def compute_transition_bid_reward(bid_steps : BidStep ): #bid_steps: [10, 4, B]
    """
        Bidding increases the risks, so the signal consists in a short-term negative reward that can lead to a bonus at the end of the game. If the current bidding value is equal to N, the *cumulative reward of the bidding team until this step* is equal to -N, and the *cumulative reward of the defending team* is eequal to 0 e.g.:
        - You bid 80 but your opponent raised to 90: reward of your team is 0, reward of other team is -90
        - You bid 80 and your teamate raised to 90: reward of your team is -90
        - You bid 80 but your opponent raised to 90. Your reward is 0. Now you raise to 100: reward is -100
        - You bid 80, everyone passed: reward is -80. You raise on yourself to 90: new reward is -10

    """
    batch_size = bid_steps.agent.shape[-1]

    def step(carry, bid_step): # bid_step : [4 x B]
        old_scores, old_ranks = carry #old_scores : [4 x B] 
        rec = nnx.vmap(bid_history_current_record)(bid_step.obs.history)
        actual_rank = rec.bid.rank.argmax(axis=-1) # 4 x B
        actual_score = actual_rank*10 + 80
        player_team_wins = (rec.author %2) == (jnp.arange(4)%2)[:,None]

        new_scores = jnp.where(player_team_wins,
                               -actual_score + old_scores, # bidding decreases the reward 
                               + old_scores) # not bidding (or gettint overbid) compensate every risk
        return (new_scores, actual_rank), new_scores

    initial_rank = jnp.full([4, batch_size], -1)
    initial_scores = jnp.zeros_like(initial_rank)

    _, bid_scores = jax.lax.scan(step, (initial_scores, initial_rank), bid_steps)
    return bid_scores



@jax.jit
def compute_final_scores (bid : BidRecord,
                          final_hist : TrickHistory, # B : final history of each game (final_hist.index = 8)
                          tricks : Trick): # 8 B
    """ Returns the final score of each team, depending whether the contract have been done or not """

    attacker_team = bid.author % 2 # shape B
    attacker_score = jax.vmap(lambda score, team : score[team])(final_hist.team_scores, attacker_team)

    contract_value = bid.bid.rank.argmax(axis=-1)*10 + 80 # shape B, between 80 and 160 (last is ALL_IN)
    has_someone_bet_p = bid.bid.rank.any(axis=-1) == True # shape B
    has_bet_all_in_p = bid.bid.rank[:,-1] # shape B
    has_made_all_in_p =  ((final_hist.tricks.best_player % 2) == attacker_team[:,None]).all(axis=-1) # shape B

    attackers_won_p = has_bet_all_in_p*has_made_all_in_p + (1-has_bet_all_in_p)*(attacker_score >= contract_value)
    # adds 250 to the attacker_score if all_in, else adds the contract value
    attackers_won_score = jax.vmap(
                            lambda team_scores, a_team, val, all_in_p: 
                                   team_scores.at[a_team].add(all_in_p*250 + (1-all_in_p)*val))(final_hist.team_scores, attacker_team, contract_value, has_bet_all_in_p)
    attackers_lose_score = jax.vmap(lambda team_scores, a_team: team_scores.at[a_team].set(0).at[~a_team].add(160))(final_hist.team_scores, attacker_team)

    team_scores = jnp.where(attackers_won_p[:,None], attackers_won_score, attackers_lose_score)
    return team_scores






