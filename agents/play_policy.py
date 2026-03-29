
from coinche.PlayState import *
from agents.trick_policy import *


def mk_playstate_step (model):
    trick_rollout = mk_rollout_episode(model)
    def playstate_step (params, playstate, key):
        """ Plays an entire trick (4 players) """
        trick_idx = playstate.current_trickstate_index

        trickstates = jax.vmap(lambda i, trick: jtu.tree_map(lambda f: f[i],trick))(trick_idx, playstate.all_trickstates)
        final_trick, trajectories = trick_rollout(trump_from_bid(playstate.bid),
                                                  trickstates,
                                                  params,
                                                  key)

        new_playstate = playstate_next_trickstate(playstate, final_trick)
        
        def compute_playstate_trajectory(trick): #trickstate in trajectory
            """ Generates 4 playstates indentical to the INITIAL playstate, but where all_trickstate[trickidx]=trickstate for each trickstate in the trajectory """
            tricksets = jtu.tree_map(lambda f,g: f.at[trick_idx].set(g), trickstates, trick) 
            return PlayState(playstate.bid, tricksets, trick_idx, playstate.current_scores)


        trajectories = jax.vmap(compute_playstate_trajectory)(trajectories)
        trajectories = jtu.tree_map(lambda f: f.swapaxes(0,1), trajectories) # [4, B, ...] => [B*4, ...]
        return new_playstate,trajectories

    return playstate_step
    #return jax.jit(playstate_step)

def mk_playstate_rollout(model):
    playstate_step = mk_playstate_step(model)

    def playstate_rollout(params, playstate, key):
        """ Simulates an entire trick game"""
        seeds = rnd.split(key, 8)
        batch_size = playstate.current_scores.shape[0]

        def scan_step (state, step_seeds):
            final_playstate, trajectory = playstate_step (params, state, step_seeds)
            return final_playstate, trajectory

        return jax.lax.scan(scan_step, playstate, seeds)

    return playstate_rollout
    #return jax.jit(playstate_rollout)



