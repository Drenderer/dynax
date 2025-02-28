"""
This file implements Pseudo Poisson Neural Networks (PPNNs)
They are basically HNNs with an arbitrary, state dependent and skew symmetric matrix 
insead of the 'symplectic' matrix. It's pseudo Poisson, because the Poisson bracket 
requirements are not enforced.
"""

import jax
import jax.numpy as jnp
import jax.nn as jnn
from jaxtyping import Scalar, Array, PRNGKeyArray

import equinox as eqx
from .._misc import jax_mod

class PseudoPoissonNN(eqx.Module):
    hamiltonian: eqx.Module
    poisson_matrix: eqx.Module
    time_dependent: bool = eqx.field(static=True)

    def __init__(self, 
                 hamiltonian: eqx.Module, 
                 poisson_matrix: eqx.Module, 
                 *,
                 time_dependent: bool = False,
                 ):

        self.hamiltonian = hamiltonian
        self.poisson_matrix = poisson_matrix
        self.time_dependent = time_dependent
        if time_dependent:
            print('WARNING: Currently only the Hamiltonian is time dependent')

    def evaluate_hamiltonian(self, x: Array, t :Scalar):
        if self.time_dependent:
            x = jnp.append(x, t)
        return self.hamiltonian(x)
    
    def evaluate_poisson_matrix(self, x: Array, t :Scalar):
        return self.poisson_matrix(x)

    def __call__(self, t:Scalar, x:Array, u:Array|None):

        if u is not None:
            raise NotImplementedError('Pseudo Poisson Neural Networks do not support inputs.')

        grad_H = jax.grad(self.evaluate_hamiltonian)(x, t)
        B = self.evaluate_poisson_matrix(x, t)
        return B @ grad_H