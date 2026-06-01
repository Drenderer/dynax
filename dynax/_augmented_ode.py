from collections.abc import Sequence
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree, Real

from ._odesolver import ODESolver
from ._pytree_utils import concat_pytree, slice_pytree

if TYPE_CHECKING:
    RealScalarLike = int | float | Array | np.ndarray
else:
    RealScalarLike = Real[ArrayLike, ""]


class AugmentedODE(eqx.Module):
    R"""Augmented ODE module.

    Extends a given initial condition with additional augmented states, then
    passes it to an [`ODESolver`][dynax.ODESolver]. Finally, the augmentation
    is removed from the solution before returning. 

    $$
    \begin{align}
        \boldsymbol{z}_0 &= \begin{bmatrix} \boldsymbol{x}_0 \\ \tilde{\boldsymbol{x}}_0 \end{bmatrix}, \\
        \dot{\boldsymbol{z}} &= \boldsymbol{f}\bigl(t, \boldsymbol{z}, \boldsymbol{u}; \mu\bigr), \\
        \boldsymbol{x}(t) &= \boldsymbol{z}(t)[:n]
    \end{align}
    $$

    with time $t \in \mathbb{R}$, state $\boldsymbol{x}(t) \in \mathbb{R}^n$, 
    augmented state $\boldsymbol{z}(t) \in \mathbb{R}^{n+n_\text{aug}}$,
    input $\boldsymbol{u}(t) \in \mathbb{R}^m$ and optional static parameters $\mu$.

    """

    ode_solver: ODESolver
    x0_auxiliary: PyTree
    concat_axes: PyTree[int] = eqx.field(static=True)

    def __init__(
        self,
        ode_solver: ODESolver,
        x0_auxiliary: PyTree,
        concat_axes: PyTree[int] = 0,
    ):
        """Initialize the ODE solver.

        Args:
            ode_solver: [`ODESolver`][dynax.ODESolver] for the augmented ODE
                `f(t, z, u, args) -> dz/dt`.
            x0_auxiliary: The initial auxiliary state, that is concatenated to
                `x0` to compute the initial augmented state. Should be a prefix
                of `x0`. To ensure that the auxiliary initial state is not
                updated during training, wrap it in `klax.non_trainable`.
                !!! note
                    Typically, your state is a 1-d array. Then `x0_auxiliary`
                    is also simply a 1-d array, containing the entries that will
                    be concatenated to `x0`.
            concat_axes: The axes along which to concatenate the state and
                auxiliary state. Should be a prefix of `x0_auxiliary`.
                !!! note


        """
        self.ode_solver = ode_solver
        self.x0_auxiliary = x0_auxiliary
        self.concat_axes = concat_axes

    def __call__(
        self,
        ts: Sequence[RealScalarLike] | Array,
        x0: PyTree,
        us: PyTree | None = None,
        args: PyTree = None,
    ) -> PyTree:
        """Solve the augmented ODE.

        Args:
            ts: Monotonic simulation time grid.
            x0: Initial (non-augmented) state at `ts[0]`.
            us: Input trajectory aligned with `ts`, or
                `None` for unforced systems.
            args: Additional arguments passed to `ODESolver.func`

        Returns:
            State trajectory sampled at `ts` with the auxiliary states removed.

        """
        z0 = concat_pytree(x0, self.x0_auxiliary, self.concat_axes)
        zs = self.ode_solver(ts, z0, us, args)
        original_sizes = jax.tree.map(
            lambda x, axis: x.shape[axis] if isinstance(x, Array) else None,
            x0,
            self.concat_axes,
        )
        xs = jax.vmap(slice_pytree, in_axes=(0, None, None, None))(
            zs, 0, original_sizes, self.concat_axes
        )
        return xs
