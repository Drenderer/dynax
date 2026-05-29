from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import diffrax
import equinox as eqx
import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Float, PyTree, Real, Scalar
from klax import NonTrainable

if TYPE_CHECKING:
    RealScalarLike = int | float | Array | np.ndarray
else:
    RealScalarLike = Real[ArrayLike, ""]

type ODEFunc = Callable[[RealScalarLike, PyTree, PyTree, PyTree], PyTree]


class ODESolver(eqx.Module):
    R"""ODE solver module.

    Solver for continuous-time ODE systems of the form

    $$
        \dot{\mathbf{x}} = \mathbf{f}\bigl(t, \mathbf{x}, \mathbf{u}; \mu\bigr)
    $$

    with time $t \in \mathbb{R}$, state $\mathbf{x}(t) \in \mathbb{R}^n$,
    input $\mathbf{u}(t) \in \mathbb{R}^m$ and optional static parameters $\mu$.

    The system is integrated using `diffrax.diffeqsolve`.
    To evaluate the input $\mathbf{u}(t)$ at arbitrary points in time Hermite
    cubic splines with backward differences are used to interpolate the discrete
    input signal $\mathbf{u}_i$ given at times $t_i$.

    Notation:
        n: State dimension.
        m: Input dimension.
        l: Number of time steps in the simulation horizon.

    """

    func: ODEFunc
    solver: diffrax.AbstractSolver
    stepsize_controller: diffrax.AbstractStepSizeController
    max_steps: int | None = eqx.field(static=True)
    dt0: Scalar | None = eqx.field(static=True)
    adjoint: diffrax.AbstractAdjoint

    def __init__(
        self,
        func: ODEFunc,
        *,
        solver: diffrax.AbstractSolver = diffrax.Tsit5(),
        stepsize_controller: diffrax.AbstractStepSizeController = diffrax.PIDController(
            rtol=1e-6, atol=1e-6
        ),
        max_steps: int | None = 4096,
        dt0: Scalar | None = None,
        adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint(),
    ):
        """Initialize the ODE solver.

        Args:
            func: Dynamics function (right hand side of the ODE)
                `f(t, x, u, args) -> dx/dt`.
            solver: Diffrax ODE solver used by `diffrax.diffeqsolve`.
            stepsize_controller: Diffrax step-size controller.
            max_steps: Maximum number of integration steps accepted by Diffrax.
                If `None`, Diffrax uses its default behavior.
            dt0: The step size to use for the first step. If using fixed step
                sizes then this will also be the step size for all other steps.
                If set as None then the initial step size will be determined
                automatically.
            adjoint: How to differentiate `diffrax.diffeqsolve`. Defaults to
                discretize-then-optimize, which is usually the best option for
                most problems. See the diffrax page on
                [Adjoints](https://docs.kidger.site/diffrax/api/adjoints/) for
                more information.

        """
        self.func = func
        self.solver = solver
        self.stepsize_controller = stepsize_controller
        self.max_steps = max_steps
        self.dt0 = dt0
        self.adjoint = adjoint

    def __call__(
        self,
        ts: Sequence[RealScalarLike] | Array,
        x0: PyTree,
        us: PyTree | None = None,
        args: PyTree = None,
    ) -> PyTree:
        """Solve the ODE.

        Args:
            ts: Monotonic simulation time grid.
            x0: Initial state at `ts[0]`.
            us: Input trajectory aligned with `ts`. If provided, cubic
                interpolation is used to evaluate $u(t)$ during
                integration. Use `None` for unforced systems.
            args: Additional arguments passed to `func`.

        Returns:
            State trajectory sampled at `ts`.

        """
        return self.get_diffrax_solution(ts, x0, us, args).ys

    def get_diffrax_solution(
        self,
        ts: Sequence[RealScalarLike] | Array,
        x0: PyTree,
        us: PyTree | None = None,
        args: PyTree = None,
    ) -> diffrax.Solution:
        """Integrate the dynamics and return the raw Diffrax solution object.

        Args:
            ts: Monotonic simulation time grid.
            x0: Initial state at `ts[0]`.
            us: Input trajectory aligned with `ts`. If provided, cubic
                interpolation is used to evaluate $u(t)$ during
                integration. Use `None` for unforced systems.
            args: Additional arguments passed to `func`.

        Returns:
            Diffrax integration result (`diffrax.Solution`) with states stored
            at sampling times `ts`.

        """
        if us is not None:
            coeffs = diffrax.backward_hermite_coefficients(ts, us)
            u_interp = diffrax.CubicInterpolation(ts, coeffs)
        else:
            u_interp = None

        def _func(t, y, args):
            u_interp, _args = args
            if u_interp is None:
                u = None
            else:
                u = u_interp.evaluate(t)

            return self.func(t, y, u, _args)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(_func),
            self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=self.dt0,
            y0=x0,
            args=(u_interp, args),
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=self.stepsize_controller,
            max_steps=self.max_steps,
            adjoint=self.adjoint,
        )

        return solution


class AugmentedODE(eqx.Module):
    R"""Augmented ODE module.

    Extends a given initial condition with additional augmented states, then
    passes it to an [`ODESolver`][dynax.ODESolver]. Finally, the augmentation
    is removed from the solution before returning. 

    $$
    \begin{align}
        \mathbf{z}_0 &= \begin{bmatrix} \mathbf{x}_0 \\ \tilde{\mathbf{x}}_0 \end{bmatrix}, \\
        \dot{\mathbf{z}} &= \mathbf{f}\bigl(t, \mathbf{z}, \mathbf{u}; \mu\bigr), \\
        \mathbf{x}(t) &= \mathbf{z}(t)[:n]
    \end{align}
    $$

    with time $t \in \mathbb{R}$, state $\mathbf{x}(t) \in \mathbb{R}^n$, 
    augmented state $\mathbf{z}(t) \in \mathbb{R}^{n+n_\text{aug}}$,
    input $\mathbf{u}(t) \in \mathbb{R}^m$ and optional static parameters $\mu$.

    """

    ode_solver: ODESolver
    augmented_x0: Array | NonTrainable[Array]
    augmented_x0_learnable: bool = eqx.field(static=True)

    def __init__(
        self,
        ode_solver: ODESolver,
        augmentation: int | Array,
        *,
        augmented_x0_learnable: bool = False,
    ):
        """Initialize the ODE solver.

        Args:
            ode_solver: [`ODESolver`][dynax.ODESolver] for the augmented ODE
                `f(t, z, u, args) -> dz/dt`.
            augmentation: If `augmentation` is an array, it describes the vector
                of augmented states that are added to the initial state `x0`
                before passing it to the `state_equation`. If augmentation is an
                integer, it describes how many extra dimensions are to be added
                to the state and the augmented initial condition is initialized
                to all zeros.
            augmented_x0_learnable: If `True`, the initial conditions of the
                augmented states are updated during training.
                Defaults to False.

        """
        self.ode_solver = ode_solver
        self.augmented_x0_learnable = augmented_x0_learnable
        if isinstance(augmentation, int):
            augmented_x0 = jnp.zeros(augmentation)
        elif eqx.is_array(augmentation):
            assert augmentation.ndim == 1, (
                "Initial condition for the augmented state must be 1-dimensional."
            )
            augmented_x0 = augmentation
        else:
            raise ValueError(
                f"'augmentation' must be an int or an array but got {augmentation}"
            )
        self.augmented_x0 = (
            augmented_x0
            if augmented_x0_learnable
            else NonTrainable(augmented_x0)
        )

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
            x0: Initial state at `ts[0]`.
            us: Input trajectory aligned with `ts`, or
                `None` for unforced systems.
            args: Additional arguments passed to `ODESolver.func`

        Returns:
            State trajectory sampled at `ts`.

        """
        z0 = jnp.concat([x0, self.augmented_x0])
        zs = self.ode_solver(ts, z0, us, args)
        xs = zs[..., : x0.size]
        return xs


class StateSpaceSystem(eqx.Module):
    R"""Module for general state-space systems.

    Module for continuous-time state-space systems of the form

    $$
    \begin{align}
    \dot{\mathbf{x}} &= \mathbf{f}\bigl(t, \mathbf{x}, \mathbf{u}; \mu\bigr), \\
    \mathbf{y} &= \mathbf{g}\bigl(t, \mathbf{x}, \mathbf{u}; \mu\bigr),
    \end{align}
    $$

    with time $t \in \mathbb{R}$, state $\mathbf{x}(t) \in \mathbb{R}^n$, 
    input $\mathbf{u}(t) \in \mathbb{R}^m$, output $\mathbf{y}(t) \in \mathbb{R}^p$ 
    and optional static parameters $\mu$. 

    The state equation is integrated using the [`ODESolver`][dynax.ODESolver] 
    module.
    """

    ode_solver: ODESolver
    output_equation: Callable[
        [Scalar, Float[Array, "n"], Float[Array, "m"], PyTree],
        Float[Array, "p"],
    ]

    def __init__(
        self,
        ode_solver: ODESolver,
        output_equation: Callable[
            [Scalar, Float[Array, "n"], Float[Array, "m"], PyTree],
            Float[Array, "p"],
        ],
    ):
        """Initialize a state-space simulation model.

        Args:
            ode_solver: [`ODESolver`][dynax.ODESolver] for the state equation.
            output_equation: Output function.

        """
        self.ode_solver = ode_solver
        self.output_equation = output_equation

    def __call__(
        self,
        ts: Sequence[RealScalarLike] | Array,
        x0: PyTree,
        us: PyTree | None = None,
        args: PyTree = None,
    ) -> PyTree:
        """Simulate the system and return output trajectories.

        Args:
            ts: Monotonic simulation time grid.
            x0: Initial state at `ts[0]`.
            us: Input trajectory aligned with `ts`, or
                `None` for unforced systems.
            args: Additional arguments passed to `ODESolver.func`

        Returns:
            Output trajectory sampled at `ts`.

        """
        xs = self.ode_solver(ts, x0, us, args)
        if us is None:
            return jax.vmap(self.output_equation, in_axes=(0, 0, None, None))(
                ts, xs, None, args
            )
        return jax.vmap(self.output_equation, in_axes=(0, 0, 0, None))(
            ts, xs, us, args
        )
