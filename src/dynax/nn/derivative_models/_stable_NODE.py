"""
This file implements stable neural ODEs
from this paper https://proceedings.neurips.cc/paper_files/paper/2019/file/0a4bbceda17a6253386bc9eb45240e25-Paper.pdf
by Kolter et al.
"""

import equinox as eqx

import jax
import jax.numpy as jnp
import jax.nn as jnn

from jaxtyping import Scalar, Array

class StableNODE(eqx.Module):
    dynamics_func: eqx.Module
    lyapunov_func: eqx.Module
    alpha: float
    input_matrix: eqx.Module

    def __init__(self,
                 dynamics_func: eqx.Module,
                 lyapunov_func: eqx.Module,
                 input_matrix: eqx.Module|None = None,
                 alpha: int = 0.
                 ):
        
        self.dynamics_func = dynamics_func
        self.lyapunov_func = lyapunov_func
        self.alpha = alpha

        self.input_matrix = input_matrix

    def __call__(self, t:Scalar|None, x:Array, u:Array|None):

        f_hat = self.dynamics_func(x)
        V, grad_V = jax.value_and_grad(self.lyapunov_func)(x)

        corr_dir = jnn.relu(jnp.inner(grad_V, f_hat) + self.alpha*V)
        norm_grad_V = jnp.inner(grad_V, grad_V)

        x_t = f_hat - corr_dir * grad_V / jnp.where(norm_grad_V==0., 1., norm_grad_V) # Avoid devision by zero in a jit tracable way
        # x_t = x_t * jnp.where(norm_grad_V==0., 0., 1.)

        # NOTE: Here I diverge from Kolters paper and include inputs in a linear way just like in ISPHS
        if u is not None:
            assert self.input_matrix is not None, 'No input matrix was provided. Cannot handle inputs.'
            g = self.input_matrix(x)
            x_t += g @ u

        return x_t
