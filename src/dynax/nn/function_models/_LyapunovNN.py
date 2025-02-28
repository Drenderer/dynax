"""
This file implements Lyapunov Neural Networks (LyapunovNN) with these properties:
    - The model is a valid Lyapunov function. It is positive (semi)-definit.
    - The Lyapunov function has no local minima.
    - The global minimum can be positioned to be at any x. 

The KolterLyapunovNN model is from this paper https://proceedings.neurips.cc/paper_files/paper/2019/file/0a4bbceda17a6253386bc9eb45240e25-Paper.pdf
by Kolter et al. The idea to use FICNNs in conjunction with INNs is also from there.
"""

import equinox as eqx

import jax
import jax.numpy as jnp
import jax.nn as jnn

from jaxtyping import Array, Scalar, PRNGKeyArray
from typing import Literal

from . import FICNN, SICNN
from ..activations import smoothed_relu, sparse_plus_shifted

class LyapunovNN(eqx.Module):
    ficnn: FICNN            # Fully input convex neural network
    inn: eqx.Module         # Invertible neural network for input warping
    minimum_learnable: bool # Determines if the minimum of the Laypunov function is learnable
    minimum: Array          # The input value (of the ficnn) for which the Lyapunov function takes on a minimum

    def __init__(self,
                ficnn: FICNN|eqx.Module,
                inn: eqx.Module = lambda x:x,
                minimum: Array|None = None,
                state_size: int|None = None,
                initialize_minimum: Literal['zeros', 'random'] = 'zeros',
                key: PRNGKeyArray|None = None,
                ):
        """Initialize a Lyapunov neural network. This is a NN from `R^n -> R` constrained 
        to be a valid Lyapunov function, suitable for showing global stability. This means 
        the resulting function has only one minimum and for every other input the function 
        is positive. (Exception: ficnn is not strictly convex at the minimum). Unless an
        optional input warping is applied, the resulting Lyapunov function is convex.

        Args:
            ficnn (FICNN | eqx.Module): Fully input convex neural network (FICNN) from `R^n -> R`
            inn (eqx.Module, optional): Optional input warping via an invertible neural network (inn) `R^n -> R^n`. Defaults to lambdax:x.
            minimum (Array | None, optional): When provided fixed the minimum location of the lyapunov function to the provided value. Defaults to None.
            state_size (int | None, optional): If ficnn is not a FICNN instance, and the minimum is not provided then the state size is needed to determine the size for the inital minimum value. Defaults to None.
            initialize_minimum (Literal['zeros', 'random'], optional): If a minimum is not provided, then this parameter determines how it is initialized. Defaults to 'zeros'.
            key (PRNGKeyArray, optional): If a minimum is not provided and should be initialized randomly, then the key parameter is necessary.

        Raises:
            ValueError: If no minimum is provided and the ficnn is not a FICNN instance. Or there are issues with the provided initialization parameters.
        """
        
        self.ficnn = ficnn
        self.inn = inn

        if state_size is None and minimum is None:
            # Try to infer the state size from the input size of the FICNN
            if isinstance(ficnn, (FICNN, SICNN)):
                state_size = ficnn.in_size
            else:
                raise ValueError('Could not determine the state size. Please supply a size for the minimum via the "state_size" argument.')

        if minimum is None:
            self.minimum_learnable = True
            if initialize_minimum == 'zeros':
                self.minimum = jnp.zeros(state_size)    # Initialize as zeros
            elif initialize_minimum == 'random':
                if key is None:
                    raise ValueError('For random initializations of the initial state a key is required.')
                initializer = jnn.initializers.normal()
                self.minimum = initializer(key, (state_size,))
            else:
                raise ValueError('initialize_minimum must be "zero" or "random".')
        else:
            self.minimum_learnable = False
            self.minimum = minimum

    def __call__(self, x: Array) -> Scalar:

        z = self.inn(x)

        if self.minimum_learnable:
            z_0 = self.inn(self.minimum)
        else:
            z_0 = self.inn(jax.lax.stop_gradient(self.minimum))

        f_0, grad_f_0 = jax.value_and_grad(self.ficnn)(z_0)
        f = self.ficnn(z)

        return f - (f_0 + jnp.inner((z - z_0), grad_f_0))




class KolterLyapunovNN(eqx.Module):
    sicnn: SICNN    # Strict input convex neural network
    inn: eqx.Module # Invertible neural network for input warping
    minimum: Array
    epsilon: float

    def __init__(self,
                 sicnn: FICNN,
                 inn: eqx.Module = eqx.nn.Identity(),
                 minimum: Array|None = None,
                 epsilon: float = 1e-3,
                 state_size: int|None = None,
                 ):
        
        if minimum is None:
            if state_size is None:
                if isinstance(sicnn, (FICNN, SICNN)):
                    state_size = sicnn.in_size
                else:
                    raise ValueError('Could not determine the state size. Please supply a size for the minimum via the "state_size" argument.')
            self.minimum = jnp.zeros(state_size)
        else:
            self.minimum = minimum

        self.sicnn = sicnn
        self.inn = inn        
        self.epsilon = epsilon

    def __call__(self, x: Array) -> Scalar:

        x = self.inn(x)
        z = self.sicnn(x) - self.sicnn(self.inn(self.minimum))
        z = smoothed_relu(z, d=0.1)
        delta = x - self.minimum
        z += self.epsilon * jnp.inner(delta, delta)
        return z
        






