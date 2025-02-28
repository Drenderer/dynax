"""
This file implements Poisson Neural Networks (PNNs)
They consist of two parts:
    1. An invertible transformation from arbitrary coordinates to
        canonical coordinates
    2. A Hamiltonian in the canonical coordinates
"""

import jax
import jax.numpy as jnp
import jax.nn as jnn
from jaxtyping import Scalar, Array, PRNGKeyArray

import equinox as eqx

class PoissonNN(eqx.Module):
    hamiltonian: eqx.Module
    transform: eqx.Module

    def __init__(self, 
                 hamiltonian: eqx.Module, 
                 transform: eqx.Module, 
                 ):

        self.hamiltonian = hamiltonian
        self.transform = transform

    def evaluate_hamiltonian(self, x: Array):
        x_c = self.transform(x)
        return self.hamiltonian(x_c)

    def __call__(self, t:Array|None, x:Array, u:None):

        if u is not None:
            raise NotImplementedError('Poisson Neural Networks do not support inputs.')

        x_c = self.transform(x)
        grad_H = jax.grad(self.hamiltonian)(x_c)

        # Apply the symplectic matrix (is faster jit-ed then matrix vector multiplication)
        a, b = jnp.split(grad_H, 2)
        x_c_dot = jnp.concat([b, -a])   

        # inv_jac = jnp.linalg.pinv(jax.jacobian(self.transform)(x))  # Pinv to be sure
        inv_jac = jax.jacobian(self.transform.inverse_call)(x_c)
        return inv_jac @ x_c_dot