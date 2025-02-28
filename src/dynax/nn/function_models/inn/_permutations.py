"""
This file implements invertible permutation layers, which should be used in combination with the affine coupling layer.
Implemented are:
- Flip layer:   Flips the input vector along its axes
- Swap layer:   Splits the input vector in two parts and swaps them
- LU layer:     Affine transformation using the LU decomposition for fast inverting. DOES NOT USE A PERMUATION MATRIX -> Can't model arbitrary affine transformations.
"""

import jax
import jax.numpy as jnp
import jax.nn as jnn

from jaxtyping import Array, PRNGKeyArray
from typing import Literal

from ._abs_invertible import InvertibleLayer


class FlipLayer(InvertibleLayer):

    def forward_call(self, x: Array) -> Array:
        return jnp.flip(x, axis=-1)
    
    def inverse_call(self, y: Array) -> Array:
        return jnp.flip(y, axis=-1)
    
    def get_jacobian_determinant(self, x: Array) -> Array:
        return 1.

class SwapLayer(InvertibleLayer):
    N: int      # Size of the input vector.
    n: int      # index of the split

    def __init__(self, N: int, n: int):
        self.N = N
        self.n = n

    def forward_call(self, x: Array) -> Array:
        x1, x2 = jnp.split(x, [self.n])
        return jnp.concat([x2, x1])
    
    def inverse_call(self, y: Array) -> Array:
        y1, y2 = jnp.split(y, [self.N - self.n])
        return jnp.concat([y2, y1])
    
    def get_jacobian_determinant(self, x: Array) -> Array:
        return 1.

class LULayer(InvertibleLayer):

    w: Array    # DxD square matrix with the components of L and U stored in one array.
    N: int      # Size of the square matrix.

    def __init__(self, N:int, *, initialize: Literal['identity', 'random'] = 'identity', key: PRNGKeyArray):
        
        if initialize == 'random':
            w_initializer = jnn.initializers.glorot_uniform()
            w = w_initializer(key, (N, N))
            P, L, U = jax.scipy.linalg.lu(w)
            self.w = L + U - jnp.eye(N)
        elif initialize == 'identity':
            self.w = jnp.eye(N)
        else: 
            raise ValueError(f'Literal "{initialize}" is not a valid value for initialize.')
        self.N = N

    def forward_call(self, x: Array) -> Array:
        L = jnp.tril(self.w, k=-1) + jnp.eye(self.N)
        U = jnp.triu(self.w)

        x = U @ x
        x = L @ x
        return x
    
    def inverse_call(self, y: Array) -> Array:
        y = jax.scipy.linalg.solve_triangular(self.w, y, lower=True, unit_diagonal=True)
        y = jax.scipy.linalg.solve_triangular(self.w, y, lower=False, unit_diagonal=False)
        return y
    
    def get_forward_jacobian(self, *args) -> Array:
        L = jnp.tril(self.w, k=-1) + jnp.eye(self.N)
        U = jnp.triu(self.w)
        return L @ U
    
    def get_jacobian_determinant(self, *args) -> Array:
        return jnp.prod(jnp.diagonal(self.w))