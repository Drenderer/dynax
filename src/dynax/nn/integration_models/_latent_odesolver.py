"""
Models of the form
`z = phi(y)`
`zs = ODESolve(ts, z0, us; f(t, z, u))`
`ys = phi^-1(zs)`
"""

import equinox as eqx
import diffrax

import jax
import jax.numpy as jnp

from jaxtyping import Array

from ._odesolver import ODESolver


class LatentODESolver(eqx.Module):
    ode_solver: ODESolver
    transform: eqx.Module   # Invertible transform

    def __init__(self,
                 derivative_model: eqx.Module,
                 transform: eqx.Module, 
                 **solver_kwargs):
        
        self.ode_solver = ODESolver(derivative_model, **solver_kwargs)
        self.transform = transform

    def __call__(self, ts:Array, y0:Array, us:Array|None) -> Array:
        """
        Given timestamps, an inital condition y0 and optional inputs,
        solve the ODE defined by `dz_dt = derivative_model(t, z, u(t))`
        in the latent space given by z = transform(y), y=transfrom.inverse_call(z).

        Args:
            ts (Array): Array of the timestamps. `shape=(num_time, )`
            y0 (Array): Array of the initial state. `shape=(state_size, )`
            us (Array | None): Array of the inputs or None if no inpus are supplied. `shape=(num_time, input_size)`

        Returns:
            Array: Solution vector. `shape=(num_time, state_size)`
        """

        z0 = self.transform.forward_call(y0)
        zs = self.ode_solver(ts, z0, us)
        ys = jax.vmap(self.transform.inverse_call)(zs)

        return ys






