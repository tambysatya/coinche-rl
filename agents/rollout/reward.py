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


def compute_transition_rewards (final_scores : Float [Array, "B 2"],

                                traj_trick_history : TrickHistory, # 8, B : Final history at the end of each trick

                                bid_steps ):
    
    def compute_bid_rewards():
        # if you bid N you lose N. If the adversary team rise above you, you earn +N (i.e. the cumulative score of all previous bid is now equal to zero)

        def step(carry, bid_step): # bid_step : [4 x B]
            old_scores, old_ranks = carry #old_scores : [4 x B] 
            rec = nnx.vmap(bid_history_current_record)(bid_step.obs.history)
            actual_rank = rec.bid.rank.argmax(axis=-1) # 4 x B


    pass

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






