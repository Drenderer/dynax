"""Lyapunov functions."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer, zeros
from jaxtyping import Array, PRNGKeyArray, Scalar

from ._misc import default_floating_dtype


class ConvexLyapunov(eqx.Module):
    R"""Convex Lyapunov normalization.

    This module normalizes a convex function by ensuring it is a valid Lyapunov
    function suitable for showing global stability. It performs the following
    transformation on a given function $f:\mathbb{R}^n \rightarrow \mathbb{R}$:
    $$ F(x) = f(x) - f(x_0) - \frac{\partial f}{\partial x} \bigg\vert_{x_0}\cdot (x-x_0) + \epsilon\left\lVert x-x_0 \right\rVert^2. $$
    This ensures the resulting function $F$ has a unique minimum at $x_0$ and 
    is positive definite due to the is a small regularization term $\epsilon$.
    """

    func: Callable[[Array], Scalar]  #: Convex function
    epsilon: float  #: Small value to ensure the Lyapunov function is positive definite
    minimum_learnable: bool  #: Determines if the minimum is learnable
    minimum: Array  #: The minimum location

    def __init__(
        self,
        func: Callable[[Array], Scalar],
        state_size: int,
        minimum_init: Initializer = zeros,
        minimum_learnable: bool = False,
        epsilon: float = 1e-6,
        dtype: type | None = None,
        *,
        key: PRNGKeyArray,
    ):
        R"""Initialize the Lyapunov normalization.

        Args:
            func: Convex function $f:\mathbb{R}^n \rightarrow \mathbb{R}$
            state_size: State size $n$ needed to determine the size for the minimum $x_0$.
            minimum_init: Initializer for the minimum location.
                Can be any function with signature `(key, shape, dtype) -> Array`
                but typically is a JAX initializer.
            minimum_learnable: If True, the minimum location $x_0$ is learnable. Otherwise,
                its gradients are stopped, preventing updates during optimization.
            epsilon: Small value to ensure the Lyapunov function is positive definite.
            dtype: The dtype to use for the minimum.
                Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.
            key: PRNG key for random initialization of the minimum.
        """
        dtype = default_floating_dtype() if dtype is None else dtype

        self.func = func
        self.minimum = minimum_init(key, (state_size,), dtype)
        self.minimum_learnable = minimum_learnable
        self.epsilon = epsilon

    def __call__(self, x: Array) -> Scalar:
        """Evaluate the Lyapunov function at a point x.

        Args:
            x: Evaluation point for the Lyapunov function.
                This should be a vector of shape `(n,)`.

        Returns:
            Value of the Lyapunov function at x.
        """
        x_0 = (
            self.minimum
            if self.minimum_learnable
            else jax.lax.stop_gradient(self.minimum)
        )

        f_0, grad_f_0 = jax.value_and_grad(self.func)(x_0)
        f = self.func(x)
        # Ensure the convex function has a minimum (at x_0)
        f_norm = f - (f_0 + jnp.inner((x - x_0), grad_f_0))
        # Add a small regularization term to ensure positive definiteness
        f_norm += self.epsilon * jnp.inner(x, x)

        return f_norm
