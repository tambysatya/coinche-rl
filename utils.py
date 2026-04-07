import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Int
from functools import partial

# from https://svn.blender.org/svnroot/bf-blender/trunk/blender/build_files/scons/tools/bcolors.py
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'




@partial(jax.jit, static_argnames=['batch_size'])
def mk_minibatches (tensor, batch_size):
    """ Transforms a tensor of size [N,...] into a tensor of size [N/B, B, ...] """
    N = tensor.shape[0]
    n_batches = N // batch_size
    batches_indices = jnp.arange(n_batches*batch_size).reshape([n_batches, batch_size])
    
    tensor = tensor[batches_indices]
    return tensor


@partial(jax.jit, static_argnames=['pool_size'])
def group_dataset_by_agent(pool_size,
                           permutation : Int [Array, "B"],
                           dataset,
                           ):
    """ Splits the dataset according to the indices:
        e.g if permutation = [0,2,2,1,1,0], the dataset will be arranged such that each row corresponds to a batch
        where each entry have the same index
        [B*P, ...] -> [P,B,....] where the pool_index of each data is specified in the permutation"""
    dataset = jtu.tree_map(lambda l: l[permutation].reshape(pool_size, l.shape[0]//pool_size, *l.shape[1:]) ,dataset)
    return dataset

@jax.jit
def ungroup_dataset_by_agent(permutation : Int[Array, "B"],
                             grouped_dataset):
    """ Undo group_dataset_by_agent """
    return jtu.tree_map(lambda l: l.reshape(l.shape[0]*l.shape[1], *l.shape[2:]).squeeze()[permutation] , grouped_dataset)


@jax.jit
def database_get (database, # batch of databases
                  position : Int [Array, "B"]):
    """ Extracts the entries at a specific (batched) index from a (batched) database.
        Input:
            database : B N Entry = batch of databases of size N
            index : B Int
        Output:
            The entry corresponding to the index
    """
    return jtu.tree_map(lambda leave:  jax.vmap(lambda l, i : l[i])(leave, position),
                database)
@jax.jit
def database_set (database,
                  new_entry,
                  position : Int [Array, "B"]):
    """ Modify the entry at a given index of a database
        Sets history_i[j] = bid[i] for j=index[i]
        input : 
            database : B N Entry = batch of database of size N 
            new_entry : B Entry = batch of entry (to be inserted)
            position : B Int = batch of positions 
    """
    return  jtu.tree_map(
                    lambda h, b: jax.vmap(
                        lambda leave, val, idx : leave.at[idx].set(val))(h, b, position),
                    database, new_entry)



