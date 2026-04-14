
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
    is_masked : Bool [Array, "B"] # If set to true, this step should not be used for training 


def mk_bid_rollout(bidding_model, pool_size):
    """ The bidding policy model outputs several logits, leading to the following decision process:
            - logit_pass : Float [Array, "B 2"] representing the probability for the player to pass
            - logit_suit : Float [Array, "B 6"] representing the probability to choose each suit (+ NO_TRUMP/ALL_TRUMP), 
            assuming the player passes
            - logit_rank : Float [Array, "B 9"] representing the probability to choose each rank (+ ALL_IN) 
    """
    graphdef, _ = nnx.split(bidding_model)
    bid_scan = mk_bid_scan(bidding_model, pool_size)

    def bid_rollout (all_params,
                     permutation,
                     hands,
                     hidden_states,
                     initial_player,
                     seed):
         batch_size = initial_player.shape[0]
         n_calls = 40  # total call = every players pass except the last one
         #n_calls = 4 * 10  # total call = every players pass except the last one
         players_order = (initial_player[:,None] + jnp.arange(4)) % 4 # starts the rotation from the initial player



         history = history_initialize(initial_player)
         bidding_count = jnp.zeros_like(initial_player, dtype=int)

         carry = hidden_states, bidding_count, history, seed
         final_carry, obs = jax.lax.scan(partial(bid_scan, all_params, players_order, permutation, hands),
                                     carry, jnp.arange(n_calls))

         hidden_states, bidding_count, history, seed = final_carry
         return hidden_states, bidding_count, history, obs
    return jax.jit(bid_rollout)

def mk_bid_scan(bidding_model, pool_size):

    graphdef, _ = nnx.split(bidding_model)
    predict_bid = mk_predict_bid(bidding_model, pool_size)

    @jax.jit
    def bid_scan (all_params,
                  players_order,
                  permutation, # to regroup everything by agent
                  hands : Int [Array, "B 4 4 8"], # 4 hands
                  carry, #4 hidden state, bidding_count, history, key
                  player_index):

         hidden_states, bidding_count, history, seed = carry
         current_player = jax.vmap (lambda order: order[player_index%4])(players_order)

         seed, key = rnd.split(seed)


         #extracts the hand and the hidden state of the current player, as well as the permutation (~ the index of the agent playing its team)
         player_hand, player_hidden, permutation = jax.vmap(lambda h,hid,p, per: (h[p], hid[p], per[p%2]))(hands, hidden_states, current_player, permutation)

         # regroup by agent [P, B/P, ...] then evaluates the batch before restoring the original order
         dataset = group_dataset_by_agent(pool_size, permutation,
                                    (current_player, player_hand, player_hidden, bidding_count, history))

         play = jax.vmap(predict_bid)(all_params, rnd.split(key, pool_size), *dataset)
         _player_hidden, _bidding_count, _history, _step = play
         player_hidden, bidding_count, history, step = ungroup_dataset_by_agent(permutation, play)

         # updates the hidden states of each player
         hidden_states = jax.vmap(lambda p,hid,player_hid:
                                                hid.at[p].set(player_hid))(
                                          current_player,
                                          hidden_states,
                                          player_hidden)

         new_carry = hidden_states, bidding_count, history, seed
         return new_carry, step
    return jax.jit(bid_scan)
                  
def mk_predict_bid(bidding_model, pool_size):

    graphdef, _ = nnx.split(bidding_model)

    def predict_bid (param,
                     key,
                     current_player : Int [Array, "B"],
                     player_hand : Hand,
                     hidden_state,
                     bidding_count : Int [Array, "B"], #number of consecutive bid that have been done (stops increasing when the bidding phase is closed)
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
                             jnp.tile(jnp.arange(9), (batch_size,1)) > rec.bid.rank.argmax(axis=-1)[:,None]) 
        logit_rank = jnp.where(bid_mask,
                        logit_rank,
                        -jnp.inf)
                               
                               

        key_pass, key_suit, key_rank = rnd.split(key, 3)
        chose_pass_p = rnd.categorical(key_pass, logit_pass).astype(bool)
        suit = rnd.categorical(key_suit, logit_suit)
        rank = rnd.categorical(key_rank, logit_rank)


        has_someone_placed_allin_p = rec.bid.rank[:,-1]
        no_raise_p = chose_pass_p | has_someone_placed_allin_p | history_is_bidding_done(history)
           
        # If the player does not or cannot raise (someone called allin), the one-hot encoding of
        # both the suit and rank are all FALSE
        new_suit = jnp.where(no_raise_p[:,None],
                             jnp.zeros([batch_size,6], dtype=bool),
                             jax.nn.one_hot(suit, 6).astype(bool))
        new_rank = jnp.where(no_raise_p[:,None],
                             jnp.zeros([batch_size,9], dtype=bool),
                             jax.nn.one_hot(rank, 9).astype(bool))
        action = Bid(new_suit, new_rank)

        logprob_pass = jnp.log(jax.nn.softmax(logit_pass))
        logprob_suit = jnp.log(jax.vmap(lambda p, s: p[s]) (jax.nn.softmax(logit_suit), suit))
        logprob_rank = jnp.log(jax.vmap(lambda p, r: p[r])(jax.nn.softmax(logit_rank), rank))


        # if the player does not raise, the history does not change
        new_history = jtu.tree_map (lambda h_pass, h_raise:
                            jnp.where( #reshape the condition dynamically
                                      no_raise_p.reshape(no_raise_p.shape + (1,)*(h_pass.ndim-no_raise_p.ndim)), 
                                      h_pass, h_raise),
                            history_player_pass(history, current_player),
                            history_player_bid(history, current_player, suit, rank))
        # if the bidding is done, the counter is stopped
        bidding_count = jnp.where (history_is_bidding_done(history),
                                   bidding_count, bidding_count+1)

        proba_pass = logprob_pass[:,1]
        proba_play = logprob_pass[:,0]+logprob_suit + logprob_rank #p(not_pass)*p(suit)*p(rank)
        branch_chose_pass = jnp.where(no_raise_p, proba_pass, proba_play)
        return (new_hidden_state,
                bidding_count,
                new_history,
                BidStep(obs,
                        action,
                        jnp.where (no_raise_p,
                                   jnp.zeros(batch_size), # log(p=1), no choice
                                   branch_chose_pass),
                        history_is_bidding_done(history)))


    return jax.jit(predict_bid)


                       

