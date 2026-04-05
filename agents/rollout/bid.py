
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
    bid : Bid
    author :  Int [Array, "B"] #who placed this bid
    checked : Bool [Array, "B 4"] #one-hot encoding of which player has already checked
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
                     current_player : Int [Array, "B"],
                     player_hand : Hand,
                     hidden_state,
                     best_bid : Bid,
                     best_player : Int [Array, "B"],
                     checked : Bool [Array, "B 4"], #who checked before
                     key):
        """ Infers a bid through the policy network."""
        batch_size = current_player.shape[0]
        obs = BidObs (hidden_state, best_bid, best_player, checked, player_hand)
        bid_policy = nnx.merge(graphdef, param)
        output, hidden_state = bid_policy(obs)
        logit_pass, logit_suit, logit_rank = output

        has_someone_placed_p = jnp.any(best_bid.rank, axis=1)
        has_someone_placed_allin_p = best_bid.rank[:,-1]
        bid_mask = jnp.where(has_someone_placed_p[:,None],
                             jnp.tile(jnp.arange(9), (batch_size,1)) > best_bid.rank.argmax(axis=1)[:,None], # forced to raise if a bid has been placed
                             jnp.ones([batch_size,9], dtype=bool))

        # masks the logit
        logit_rank = jnp.where(bid_mask,
                        logit_rank,
                        -jnp.inf)
                               
                               

        key_pass, key_suit, key_rank = rnd.split(key, 3)
        pass_p = rnd.categorical(key_pass, logit_pass).astype(bool)
        suit = rnd.categorical(key_suit, logit_suit)
        rank = rnd.categorical(key_rank, logit_rank)

        # empty bid if the player "pass" or if the player cannot raise (e.g. all_in somewhere)
        pass_bid = Bid(jnp.zeros([batch_size,6], dtype=bool),
                       jnp.zeros([batch_size,9], dtype=bool),
                       current_player)

        # the bid sampled by the network
        raise_bid = Bid(jax.nn.one_hot(suit,6).astype(bool),
                        jax.nn.one_hot(rank,9).astype(bool),
                        current_player)

        has_to_pass_p = pass_p | has_someone_placed_allin_p
        new_bid = jtu.tree_map(lambda pass_b, raise_b:
                                    jnp.where (has_to_pass_p[:,None],
                                               pass_b, raise_b),
                                pass_bid, raise_bid)
        logprob_pass = jnp.log(jax.nn.softmax(logit_pass))
        logprob_suit = jnp.log(jax.nn.softmax(logit_suit))
        logprob_rank = jnp.log(jax.nn.softmax(logit_rank))
        return new_bid, BidStep(obs,
                            new_bid, 
                            jnp.where (has_someone_placed_p,
                                 jnp.zeros([batch_size]), #log (1)
                                 jnp.where (pass_p,
                                    logprob_pass[:,1],
                                    logprob_pass[:,0]+logprob_suit[:,suit] + logprob_rank[:,rank]))) # p(not_pass)*p(suit)*p(rank)


    return jax.jit(predict_bid)


                       

