"""
This file implements the affine coupling layer described by 
[1] L. Dinh, J. Sohl-Dickstein, and S. Bengio, "Density estimation using Real NVP". arXiv, 2016. doi: 10.48550/ARXIV.1605.08803. Available: https://arxiv.org/abs/1605.08803

Using a custom MLP initialization the inital weights are chosen 10x smaller than usual and all biases are initially 0.
This way the affine coupling layer approximates the identity map initially.
"""

import equinox as eqx

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr

from jaxtyping import Array, PRNGKeyArray
from typing import Union, Literal, Callable

from ._abs_invertible import InvertibleLayer

def custom_mlp_initilization(mlp: eqx.nn.MLP) -> eqx.nn.MLP:

    biases  = lambda m: [l.bias for l in m.layers]
    weights = lambda m: m.layers[-1].weight #[l.weight for l in m.layers] 
    set_zero = lambda x: jnp.zeros_like(x)
    reduce = lambda x: x*1e-1  # Reduce by a factor of ten, but don't set to 0!
    mlp = eqx.tree_at(biases, mlp, replace_fn=set_zero)
    mlp = eqx.tree_at(weights, mlp, replace_fn=set_zero)
    return mlp


class AffineCouplingLayer(InvertibleLayer):
    n: int  # Vector splitting index
    N: int  # Input and output dimension
    scale: eqx.nn.MLP
    translation: eqx.nn.MLP

    def __init__(self, 
                 N: int, 
                 n: int, 
                 width: int=16, 
                 depth: int=2, 
                 activation: Callable=jnn.sparse_plus, 
                 final_activation: Callable=lambda x: x,
                 *, 
                 initialize: Literal['identity', 'random'] = 'identity', 
                 key: PRNGKeyArray):

        assert 0 < n < N, 'Input must be split into two sections with at least one element each (0<n<N).'

        self.n = n
        self.N = N

        size1, size2 = n, N-n

        key_s, key_t = jr.split(key, 2)
        scale       = eqx.nn.MLP(size1, size2, width, depth, activation, final_activation, key=key_s)
        translation = eqx.nn.MLP(size1, size2, width, depth, activation, final_activation, key=key_t)

        if initialize == 'identity':
            self.scale       = custom_mlp_initilization(scale)
            self.translation = custom_mlp_initilization(translation)
        elif initialize == 'random':
            self.scale       = scale
            self.translation = translation
        else:
            raise ValueError(f'Literal "{initialize}" is not a valid value for initialize.')

    def forward_call(self, x: Array) -> Array:

        x1, x2 = jnp.split(x, [self.n])

        y1 = x1
        y2 = x2 * jnp.exp(self.scale(x1)) + self.translation(x1)

        return jnp.concat([y1, y2])

    def inverse_call(self, y: Array) -> Array:

        y1, y2 = jnp.split(y, [self.n])
        
        x1 = y1
        x2 = (y2 - self.translation(y1)) * jnp.exp(-self.scale(y1)) 

        return jnp.concat([x1, x2])
    
