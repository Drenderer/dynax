"""
This file implements the affine coupling block as described by
[2] L. Ardizzone et al., "Analyzing Inverse Problems with Invertible Neural Networks". arXiv, Feb. 06, 2019. Accessed: Jul. 08, 2024. [Online]. Available: http://arxiv.org/abs/1808.04730

It is simply the sequence: Swap-, AffineCoupling-, Swap- and AffineCouplingLayer
"""

import jax.nn as jnn
import jax.random as jr

from jaxtyping import Array, PRNGKeyArray
from typing import Literal, Callable

from ._affine_coupling_layer import AffineCouplingLayer
from ._permutations import SwapLayer
from ._abs_invertible import InvertibleLayer


class AffineCouplingBlock(InvertibleLayer):
    n: int
    N: int
    layers: list

    def __init__(self, 
                 N: int, 
                 n: int, 
                 width: int=16, 
                 depth: int=2, 
                 activation: Callable=jnn.sparse_plus, 
                 final_activation: Callable=lambda x:x,
                 *, 
                 initialize: Literal['identity', 'random'] = 'identity', 
                 key: PRNGKeyArray):

        assert 0 < n < N, 'Input must be split into two sections with at least one element each (0<n<N).'

        self.n = n
        self.N = N
        d = N - n

        key1, key2 = jr.split(key, 2)

        self.layers = [
            SwapLayer(N, n),
            AffineCouplingLayer(N, d, width, depth, activation, final_activation, initialize=initialize, key=key1),
            SwapLayer(N, d),
            AffineCouplingLayer(N, n, width, depth, activation, final_activation, initialize=initialize, key=key2),
        ]

    def forward_call(self, x: Array) -> Array:

        for l in self.layers:
            x = l.forward_call(x)
        return x
    
    def inverse_call(self, y: Array) -> Array:

        for l in self.layers[::-1]:
            y = l.inverse_call(y)
        return y
