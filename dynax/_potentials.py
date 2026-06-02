"""Lyapunov functions."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer, zeros
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar
from klax import NonTrainable

from ._misc import default_floating_dtype


class ConvexNormalization(eqx.Module):
    R"""Normalization for convex scalar potentials.

    This module normalizes a given function
    $f:\mathbb{R}^n \times \mathcal{P} \rightarrow \mathbb{R},\, (x; \text{args}) \mapsto f(x; \text{args})$
    convex in $x$, by applying the following transformation:
    $$
        F(x; \text{args}) = f(x; \text{args}) - f(x^\ast; \text{args}) - \frac{\partial f}{\partial x} \bigg\vert_{x^\ast}\cdot (x-x^\ast) + \epsilon\left\lVert x-x^\ast \right\rVert^2.
    $$
    with $x^\ast \in \mathbb{R}^n$ and $\epsilon \geq 0$.
    This ensures the resulting function $F(x; \text{args})$ has the following properties,
    given $f$ is convex in $x$:

    - $F$ is convex in $x$,
    - $F$ is strictly convex in $x$ if $\epsilon>0$,
    - $F$ has a unique minimum at $x^\ast$ and,
    - $F(x^\ast) = 0$.

    The intended use of this module is to normalize a (partially) convex neural network
    and use the resulting function as a Lyapunov candidate function, or as
    scalar potential to ensure global Lyapunov stability by construction.
    """

    func: Callable[[Array, PyTree], Scalar]
    epsilon: float
    argmin_learnable: bool
    argmin: Array

    def __init__(
        self,
        func: Callable[[Array, PyTree], Scalar],
        state_size: int,
        argmin_init: Initializer = zeros,
        argmin_learnable: bool = False,
        epsilon: float = 1e-6,
        dtype: type | None = None,
        *,
        key: PRNGKeyArray,
    ):
        R"""Initialize the Lyapunov normalization.

        Args:
            func: Function `f(x, args) -> Scalar`, convex in `x`.
            state_size: State size `n` needed to determine the size for the
                minimizer `x*`.
            argmin_init: JAX initializer for the minimum location `x*`.
                Can be any function with signature `(key, shape, dtype) -> Array`
                but typically is a JAX initializer.
            argmin_learnable: If True, the minimizer `x*` is learnable.
                Otherwise, it is wrapped in `klax.NonTrainable`, preventing
                updates during optimization.
            epsilon: Small value to ensure the resulting function is strictly
                convex.
                !!! warning
                    If `epsilon` is an inexact array
                    (`eqx.is_inexact_array(epsilon)) == True`) then it will
                    receive gradient updates. Make sure to use a plain python
                    float or wrap it in `klax.non_trainable`.
            dtype: The dtype to use for the minimum.
                Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.
            key: PRNG key for random initialization of the minimum.

        """
        dtype = default_floating_dtype() if dtype is None else dtype

        self.func = func
        argmin = argmin_init(key, (state_size,), dtype)
        self.argmin = argmin if argmin_learnable else NonTrainable(argmin)
        self.argmin_learnable = argmin_learnable
        self.epsilon = epsilon

    def __call__(self, x: Array, args: PyTree) -> Scalar:
        """Evaluate the normalized function.

        Args:
            x: Evaluation point for the normalized function.
                This should be a vector of shape `(n,)`.
            args: Additional arguments passed to the wrapped function.

        Returns:
            Value of the normalized function at `x, args`.

        """
        f_star, grad_f_star = jax.value_and_grad(self.func)(self.argmin, args)
        f = self.func(x, args)
        dx = x - self.argmin
        # Ensure the convex function has a minimum (at x_star)
        f_norm = f - (f_star + jnp.inner(dx, grad_f_star))
        # Add a small regularization term to ensure strict convexity
        f_norm += self.epsilon * jnp.inner(dx, dx)

        return f_norm


class NoArgsPotential(eqx.Module):
    """Wraps a callable `x -> Scalar` for use with [`ConvexNormalization`][dynax.ConvexNormalization].

    Convenience wrapper that just adds the `args` argument to the call
    signature to ensure compatibility.

    Example:
        ```python
        func = klax.nn.FICNN(4, "scalar", ...)
        potential = ConvexNormalization(NoArgsPotential(func), state_size=4)
        ```

    """

    func: Callable[[Array], Scalar]

    def __call__(self, x: Array, args: PyTree) -> Scalar:
        x_flat, _ = jax.flatten_util.ravel_pytree(x)
        return self.func(x_flat)
