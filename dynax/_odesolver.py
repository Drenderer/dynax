from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import diffrax
import equinox as eqx
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree, Real, Scalar

if TYPE_CHECKING:
    RealScalarLike = int | float | Array | np.ndarray
else:
    RealScalarLike = Real[ArrayLike, ""]

type ODEFunc = Callable[[RealScalarLike, PyTree, PyTree, PyTree], PyTree]


class ODESolver(eqx.Module):
    R"""ODE solver module.

    Solver for continuous-time ODE systems of the form

    $$
        \dot{\boldsymbol{x}} = \boldsymbol{f}\bigl(t, \boldsymbol{x}, \boldsymbol{u}; \text{args}\bigr)
    $$

    with time $t \in \mathbb{R}$, state $\boldsymbol{x}(t) \in \mathbb{R}^n$,
    input $\boldsymbol{u}(t) \in \mathbb{R}^m$ and optional static parameters $\text{args}$.

    The system is integrated using `diffrax.diffeqsolve`.
    To evaluate the input $\boldsymbol{u}(t)$ at arbitrary points in time Hermite
    cubic splines with backward differences are used to interpolate the discrete
    input signal $\boldsymbol{u}_i$ given at times $t_i$.

    Notation:
        n: State dimension.
        m: Input dimension.
        l: Number of time steps in the simulation horizon.

    """

    func: ODEFunc
    solver: diffrax.AbstractSolver
    stepsize_controller: diffrax.AbstractStepSizeController
    max_steps: int | None
    dt0: Scalar | None
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
