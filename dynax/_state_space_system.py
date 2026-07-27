from collections.abc import Callable

import equinox as eqx
import jax
from jaxtyping import Array, PyTree

from ._odesolver import ODESolver
from ._typing import RealScalarLike


class StateSpaceSystem(eqx.Module):
    R"""Module for general state-space systems.

    Module for continuous-time state-space systems of the form

    $$
    \begin{align}
    \dot{\boldsymbol{x}} &= \boldsymbol{f}\bigl(t, \boldsymbol{x}, \boldsymbol{u}; \text{args}\bigr), \\
    \boldsymbol{y} &= \boldsymbol{g}\bigl(t, \boldsymbol{x}, \boldsymbol{u}; \text{args}\bigr),
    \end{align}
    $$

    with time $t \in \mathbb{R}$, state $\boldsymbol{x}(t) \in \mathbb{R}^n$,
    input $\boldsymbol{u}(t) \in \mathbb{R}^m$, output $\boldsymbol{y}(t) \in \mathbb{R}^p$
    and optional static parameters $\text{args}$.

    The state equation is integrated using the [`ODESolver`][dynax.ODESolver]
    module.
    """

    ode_solver: ODESolver
    output_equation: Callable[[RealScalarLike, PyTree, PyTree, PyTree], PyTree]

    def __init__(
        self,
        ode_solver: ODESolver,
        output_equation: Callable[
            [RealScalarLike, PyTree, PyTree, PyTree], PyTree
        ],
    ):
        """Initialize a state-space simulation model.

        Args:
            ode_solver: [`ODESolver`][dynax.ODESolver] for the state equation.
            output_equation: Output function `g(t, x, u, args) -> y`.

        """
        self.ode_solver = ode_solver
        self.output_equation = output_equation

    def __call__(
        self,
        ts: Array,
        x0: PyTree,
        us: PyTree | None = None,
        args: PyTree = None,
    ) -> PyTree:
        """Simulate the system and return output trajectories.

        Args:
            ts: Monotonic simulation time grid.
            x0: Initial state at `ts[0]`.
            us: Input trajectory aligned with `ts`, or `None` for unforced
                systems. Each leaf is expected to have a leading time axis
                of length `len(ts)`. Passed sample-wise to `output_equation`
                (i.e. `u(ts[i])`, not interpolated).
            args: Additional arguments passed to `output_equation`.

        Returns:
            Output trajectory sampled at `ts`.

        """
        xs = self.ode_solver(ts, x0, us, args)
        us_in_axes = jax.tree.map(lambda _: 0, us) if us is not None else None
        return jax.vmap(
            self.output_equation, in_axes=(0, 0, us_in_axes, None)
        )(ts, xs, us, args)
