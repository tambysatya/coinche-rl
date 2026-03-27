import jax
import jax.numpy as jnp
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



