"""
In this file implements monotonic neural networks.
Each element in the output is a monotonic function of each input element.
"""

import equinox as eqx

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr

from jaxtyping import Array, PRNGKeyArray
from typing import Union, Literal, Callable


from ..constraints import NonNegative

class MonotonicNN(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, 
                 in_size: Union[int, Literal['scalar']], 
                 out_size: Union[int, Literal['scalar']], 
                 width: int,
                 depth: int, 
                 activation: Callable = jnn.tanh, 
                 *, 
                 key: PRNGKeyArray):

        mlp = eqx.nn.MLP(in_size, out_size, width, depth, activation, key=key)

        is_linear = lambda x: isinstance(x, eqx.nn.Linear)
        get_weights = lambda m: [x.weight
                                 for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                                 if is_linear(x)]
        mlp = eqx.tree_at(get_weights, mlp, replace_fn=lambda x: jnp.eye(*x.shape))
        self.mlp = eqx.tree_at(get_weights, mlp, replace_fn=NonNegative)

    def __call__(self, x):
        return self.mlp(x)


def get_monotonicNN(in_size: Union[int, Literal['scalar']], 
                    out_size: Union[int, Literal['scalar']], 
                    width: int,
                    depth: int, 
                    activation: Callable = jnn.tanh, 
                    *, 
                    key: PRNGKeyArray):
    
    mlp = eqx.nn.MLP(in_size, out_size, width, depth, activation, key=key)

    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [x.weight
                                for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                                if is_linear(x)]
    mlp = eqx.tree_at(get_weights, mlp, replace_fn=lambda x: jnp.eye(*x.shape))
    return eqx.tree_at(get_weights, mlp, replace_fn=NonNegative)

