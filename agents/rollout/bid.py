
from flax import struct
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as rnd
import jax.tree_util as jtu
import flax.nnx as nnx


from coinche.Bid import *
from coinche.Hand import *


@struct.dataclass
class BidObs:
    """A partial observation of a bid"""
    hidden_state : jax.Array
    history : BidHistory
    hand : Hand


@struct.dataclass
class BidStep:
    """ Experience collected during the rollouts """
    obs : BidObs
    action : Bid
    logprobs : Float [Array, "B"]


def mk_bid_rollout(bidding_model, pool_size):
    """ The bidding policy model outputs several logits, leading to the following decision process:
            - logit_pass : Float [Array, "B 2"] representing the probability for the player to pass
            - logit_suit : Float [Array, "B 6"] representing the probability to choose each suit (+ NO_TRUMP/ALL_TRUMP), 
            assuming the player passes
            - logit_rank : Float [Array, "B 9"] representing the probability to choose each rank (+ ALL_IN) 
    """
    graphdef, _ = nnx.split(bidding_model)

                 
                  
    @jax.jit
    def predict_bid (param,
                     key,
                     current_player : Int [Array, "B"],
                     player_hand : Hand,
                     hidden_state,
                     history : BidHistory
                     ):
        """ Infers a bid through the policy network."""
        batch_size = current_player.shape[0]
        obs = BidObs (hidden_state, history, player_hand)
        bid_policy = nnx.merge(graphdef, param)
        output, new_hidden_state = bid_policy(obs)
        logit_pass, logit_suit, logit_rank = output

        rec = history_current_record(history)


        # If a bid have been placed, all possible bids have a higher rank
        has_someone_placed_p = history_is_empty(history)
        bid_mask = jnp.where(has_someone_placed_p[:,None],
                             jnp.ones([batch_size,9], dtype=bool),
                             jnp.tile(jnp.arange(9), (batch_size,1)) > rec.bid.rank.argmax(axis=1)[:,None]) 
        logit_rank = jnp.where(bid_mask,
                        logit_rank,
                        -jnp.inf)
                               
                               

        key_pass, key_suit, key_rank = rnd.split(key, 3)
        pass_p = rnd.categorical(key_pass, logit_pass).astype(bool)
        suit = rnd.categorical(key_suit, logit_suit)
        rank = rnd.categorical(key_rank, logit_rank)


        has_someone_placed_allin_p = rec.bid.rank[:,-1]
        has_to_pass_p = pass_p | has_someone_placed_allin_p
           
        # If the player does not or cannot raise (someone called allin), the one-hot encoding of
        # both the suit and rank are all FALSE
        new_suit = jnp.where(has_to_pass_p[:,None],
                             jnp.zeros([batch_size,6], dtype=bool),
                             jax.nn.one_hot(suit, 6).astype(bool))
        new_rank = jnp.where(has_to_pass_p[:,None],
                             jnp.zeros([batch_size,9], dtype=bool),
                             jax.nn.one_hot(rank, 9).astype(bool))
        action = Bid(new_suit, new_rank)

        logprob_pass = jnp.log(jax.nn.softmax(logit_pass))
        logprob_suit = jnp.log(jax.nn.softmax(logit_suit))
        logprob_rank = jnp.log(jax.nn.softmax(logit_rank))
        new_history = jtu.tree_map (lambda h_pass, h_raise:
                            jnp.where(pass_p[:,None,None], h_pass, h_raise),
                            history_player_pass(history, current_player),
                            history_player_bid(history, current_player, suit, rank))

        return (new_hidden_state,
                action,
                BidStep(obs,
                        action,
                        jnp.where (has_someone_placed_p,
                                   jnp.zeros([batch_size]), #log (1)
                                   jnp.where (pass_p,
                                              logprob_pass[:,1],
                                              logprob_pass[:,0]+logprob_suit[:,suit] + logprob_rank[:,rank])))) # p(not_pass)*p(suit)*p(rank)


    return predict_bid


                       

