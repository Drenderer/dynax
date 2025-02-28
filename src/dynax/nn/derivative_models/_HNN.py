"""
This file implements Hamiltonian Neural Networks (HNNs)
from this paper https://proceedings.neurips.cc/paper_files/paper/2019/file/26cd8ecadce0d4efd6cc8a8725cbd1f8-Paper.pdf
by Greydanus et al.
"""

import jax
import jax.numpy as jnp
import jax.nn as jnn
from jaxtyping import Scalar, Array, PRNGKeyArray

import equinox as eqx
from .._misc import jax_mod

class  HamiltonianNN(eqx.Module):
    submodel: eqx.Module
    J: Array
    time_dependent: bool = eqx.field(static=True)
    periods: Array

    def __init__(self, 
                 submodel: eqx.Module, 
                 state_size: int,
                 *,
                 time_dependent: bool = False,
                 periods: Array|None = None):

        assert state_size%2==0, 'The state_site must be an even integer! (Position and momentum for every degree of freedom)'
        self.submodel = submodel

        n_dof = state_size//2
        self.J = jnp.block([[jnp.zeros((n_dof, n_dof)),  jnp.eye(n_dof)], 
                            [-jnp.eye(n_dof), jnp.zeros((n_dof, n_dof))]])
        
        self.time_dependent = time_dependent

        if periods is None:
            self.periods = jnp.zeros(state_size)
        else:
            assert periods.shape == (n_dof,), 'Periods must be specified for each degree of freedom and shoud have a size of half the state_size.'
            self.periods = jnp.concat([periods, jnp.zeros(n_dof)], axis=0)

    def hamiltonian(self, q:Array, p:Array, t:Scalar|None=None):
        assert jnp.shape(q) == jnp.shape(p), "p and q must have the same shape"
        # Combine generalized positions and momentum to the state vector x.
        q = jnp.atleast_1d(q)
        p = jnp.atleast_1d(p)
        
        x = jnp.concat([q, p], axis=0)    
        return self._hamiltonian(x, t)

    def _hamiltonian(self, x: Array, t :Scalar):
        x = jax_mod(x, jax.lax.stop_gradient(self.periods))
        if self.time_dependent:
            x = jnp.append(x, t)
        return self.submodel(x)

    def __call__(self, t:Scalar, x:Array, u:Array|None):

        if u is not None:
            raise NotImplementedError('Hamiltonian Neural Networks do not support inputs.')

        grad_H = jax.grad(self._hamiltonian)(x, t)
        return jax.lax.stop_gradient(self.J) @ grad_H