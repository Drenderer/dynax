from collections.abc import Callable

import diffrax
import equinox as eqx
import jax
from jaxtyping import Array, Float, PyTree, Scalar

type StateEquation = Callable[
    [Scalar, Float[Array, "n"], Float[Array, "m"], PyTree],
    Float[Array, "n"],
]

type OutputEquation = Callable[
    [Scalar, Float[Array, "n"], Float[Array, "m"], PyTree],
    Float[Array, "p"],
]


class FullStateOutput(eqx.Module):
    """Output map that returns the full state as output.

    This corresponds to $y(t)=x(t)$, i.e., output dimension equals state
    dimension ($p=n$).
    """

    def __call__(
        self,
        t: Scalar,
        x: Float[Array, "n"],
        u: Float[Array, "m"],
        args: PyTree,
    ) -> Float[Array, "n"]:
        """Return the state unchanged.

        Args:
            t: Scalar time value (`Scalar`). Unused.
            x: State vector (`Float[Array, "n"]`).
            u: Input vector (`Float[Array, "m"] | None`). Unused.
            args: Additional arguments (`PyTree`). Unused.

        Returns:
            State `x` (`Float[Array, "n"]`).

        """
        return x


class StateSpaceSystem(eqx.Module):
    R"""ODE solver class for general state-space systems.

    Generic continuous-time state-space system interface for systems of the form

    $$
    \begin{aligned}
    \dot{x}(t) &= f\bigl(t, x(t), u(t), \mu\bigr), \\
    y(t) &= g\bigl(t, x(t), u(t), \mu\bigr),
    \end{aligned}
    $$

    with state $x \in \mathbb{R}^n$, input $u \in \mathbb{R}^m$, output
    $y \in \mathbb{R}^p$ and optional static parameters $\mu$. 
    The system is integrated using `diffrax.diffeqsolve`.

    Notation:
        n: State dimension symbol used in shape annotations, e.g.
            `Float[Array, "n"]`.
        m: Input dimension symbol used in shape annotations, e.g.
            `Float[Array, "m"]`.
        p: Output dimension symbol used in shape annotations, e.g.
            `Float[Array, "p"]`.
        l: Number of time steps in the simulation horizon, used in shape
            annotations such as `Float[Array, "l n"]`.

    Example:
        Minimal usage with an unforced linear system (`u=None`):

        ```python
        import jax.numpy as jnp
        from dynax._state_space_system import StateSpaceSystem

        A = jnp.array([[0.0, 1.0], [-1.0, -0.1]])


        def f(t, x, u, params):
            return A @ x


        system = StateSpaceSystem(state_equation=f)
        ts = jnp.linspace(0.0, 5.0, 200)
        x0 = jnp.array([1.0, 0.0])

        ys = system(ts=ts, x0=x0, us=None, params={"tag": "metadata"})
    ```

    """

    state_equation: StateEquation
    output_equation: OutputEquation
    solver: diffrax.AbstractSolver
    stepsize_controller: diffrax.AbstractStepSizeController
    max_steps: int | None = eqx.field(static=True)

    def __init__(
        self,
        state_equation: StateEquation,
        output_equation: OutputEquation = FullStateOutput(),
        solver: diffrax.AbstractSolver = diffrax.Tsit5(),
        stepsize_controller: diffrax.AbstractStepSizeController = diffrax.PIDController(
            rtol=1e-6, atol=1e-6
        ),
        max_steps: int | None = 4096,
    ):
        """Initialize a state-space simulation model.

        Args:
            state_equation: Dynamics function (right hand side of an ODE).
            output_equation: Output function.
            solver: Diffrax ODE solver used by `diffrax.diffeqsolve`.
            stepsize_controller: Diffrax step-size controller.
            max_steps: Maximum number of integration steps accepted by Diffrax.
                If `None`, Diffrax uses its default behavior.

        Notes:
            `params` is forwarded to both `state_equation` and `output_equation`
            during simulation and can be any PyTree.

        """
        self.state_equation = state_equation
        self.output_equation = output_equation
        self.solver = solver
        self.stepsize_controller = stepsize_controller
        self.max_steps = max_steps

    def __call__(
        self,
        ts: Float[Array, "l"],
        x0: Float[Array, "n"],
        us: Float[Array, "l m"] | None = None,
        params: PyTree = None,
    ) -> Float[Array, "l p"]:
        """Simulate the system and return output trajectories.

        Args:
            ts: Monotonic simulation time grid (`Float[Array, "l"]`).
            x0: Initial state at `ts[0]` (`Float[Array, "n"]`).
            us: Input trajectory (`Float[Array, "l m"]`) aligned with `ts`, or
                `None` for unforced systems.
            params: Additional arguments (`PyTree`) forwarded to both
                `state_equation` and `output_equation`.

        Returns:
            Output trajectory sampled at `ts` (`Float[Array, "l p"]`).

        """
        xs = self.solve(ts, x0, us, params)
        if us is None:
            return jax.vmap(self.output_equation, in_axes=(0, 0, None, None))(
                ts, xs, None, params
            )
        return jax.vmap(self.output_equation, in_axes=(0, 0, 0, None))(
            ts, xs, us, params
        )

    def solve(
        self,
        ts: Float[Array, "l"],
        x0: Float[Array, "n"],
        us: Float[Array, "l m"] | None = None,
        params: PyTree = None,
    ) -> Float[Array, "l n"]:
        """Simulate the system and return state trajectories.

        Args:
            ts: Monotonic simulation time grid (`Float[Array, "l"]`).
            x0: Initial state at `ts[0]` (`Float[Array, "n"]`).
            us: Input trajectory (`Float[Array, "l m"]`) aligned with `ts`, or
                `None` for unforced systems.
            params: Additional arguments (`PyTree`) forwarded to the
                `state_equation`. This is not a learnable parameter container.

        Returns:
            State trajectory sampled at `ts` (`Float[Array, "l n"]`).

        """
        return self.get_diffrax_solution(ts, x0, us, params).ys

    def get_diffrax_solution(
        self,
        ts: Float[Array, "l"],
        x0: Float[Array, "n"],
        us: Float[Array, "l m"] | None = None,
        params: PyTree = None,
    ) -> diffrax.Solution:
        """Integrate the dynamics and return the raw Diffrax solution object.

        Args:
            ts: Monotonic simulation time grid (`Float[Array, "l"]`).
            x0: Initial state at `ts[0]` (`Float[Array, "n"]`).
            us: Input trajectory (`Float[Array, "l m"]`) aligned with `ts`. If
                provided, cubic interpolation is used to evaluate $u(t)$ during
                integration. Use `None` for unforced systems.
            params: Additional arguments (`PyTree`) passed to `state_equation`
                via Diffrax `args`. This is not a learnable parameter container.

        Returns:
            Diffrax integration result (`diffrax.Solution`) with states stored at
            the requested sampling times `ts`.

        """
        if us is not None:
            coeffs = diffrax.backward_hermite_coefficients(ts, us)
            u_interp = diffrax.CubicInterpolation(ts, coeffs)
        else:
            u_interp = None

        def _func(t, y, args):
            u_interp, params = args
            if u_interp is None:
                u = None
            else:
                u = u_interp.evaluate(t)

            return self.state_equation(t, y, u, params)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(_func),
            self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=x0,
            args=(u_interp, params),
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=self.stepsize_controller,
            max_steps=self.max_steps,
        )

        return solution
