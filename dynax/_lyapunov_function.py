"""
This file implements Lyapunov Neural Networks (LyapunovNN) with these properties:
    - The model is a valid Lyapunov function. It is positive (semi)-definit scalar function.
    - The Lyapunov function has no local minima.
    - The global minimum can be positioned to be at any x.
"""

import equinox as eqx
from ._misc import default_floating_dtype

import jax
from jax.nn.initializers import Initializer, zeros
import jax.numpy as jnp

from jaxtyping import Array, Scalar, PRNGKeyArray
from typing import Callable


class LyapunovNN(eqx.Module):
    """
    Lyapunov neural network. This is a NN from `R^n -> R` constrained
    to be a valid Lyapunov function, suitable for ensuring global stability. This means
    the resulting function has only one minimum and for every other input the function
    is positive.
    """
    ficnn: Callable[[Array], Scalar]  #: Fully input convex neural network
    epsilon: float  #: Small value to ensure the Lyapunov function is positive definite
    minimum_learnable: bool  #: Determines if the minimum is learnable
    minimum: Array  #: The minimum location

    def __init__(
        self,
        ficnn: Callable[[Array], Scalar],
        state_size: int,
        minimum_init: Initializer = zeros,
        minimum_learnable: bool = False,
        epsilon: float = 1e-6,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """

        Args:
            ficnn: Convex function from ``R^n -> R``
            minimum_init: Initializer for the minimum location.
                Can be any function with signature ``(key, shape, dtype) -> Array``
                but typically is a JAX initializer.
                Defaults to ``zeros``.
            state_size: If ficnn does not have a ``in_size`` attribute, then the state size is
                needed to determine the size for the inital minimum.
                Defaults to None.
            epsilon: Small value to ensure the Lyapunov function is positive definite.
                A regulatization term epsilon*|x|^2 is added to the Lyapunov function.
                Defaults to 1e-6.
            key: PRNG key for random initialization of the minimum.
                Defaults to None.

        Raises:
            ValueError: If no minimum is provided and the state size could not be infered from
                the in_size attribute of ``ficnn``.
                Or there are issues with the provided initialization parameters.
        """

        dtype = default_floating_dtype() if dtype is None else dtype

        self.ficnn = ficnn
        self.minimum = minimum_init(key, (state_size,), dtype)
        self.minimum_learnable = minimum_learnable
        self.epsilon = epsilon

    def __call__(self, x: Array) -> Scalar:
        """
        Args:
            x: Evaluation point for the Lyapunov function.
                This should be a vector of shape (state_size,).
        
        Returns:
            Value of the Lyapunov function at x.
        """
        x_0 = (
            self.minimum
            if self.minimum_learnable
            else jax.lax.stop_gradient(self.minimum)
        )

        f_0, grad_f_0 = jax.value_and_grad(self.ficnn)(x_0)
        f = self.ficnn(x)
        # Ensure the convex function has a minimum (at x_0)
        f_norm = f - (f_0 + jnp.inner((x - x_0), grad_f_0))
        # Add a small regularization term to ensure positive definiteness
        f_norm += self.epsilon * jnp.inner(x, x)

        return f_norm
