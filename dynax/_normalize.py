from typing import TYPE_CHECKING, Literal, TypeVar, cast

from jax import numpy as jnp
from jaxtyping import Array, Float
from scipy.optimize import OptimizeResult, minimize_scalar

# TODO: Add tests!
# TODO: Add more userfriendly normalization

type Scalar = Float[Array, "1"]


def normalization_coefficients(
    std_y: Float[Array, "n"] | float,
    std_v: Float[Array, "n"] | float | None = None,
    std_a: Float[Array, "n"] | float | None = None,
    w_y: float = 1.0,
    w_v: float = 1.0,
    w_a: float = 1.0,
    tol: float = 1e-6,
    maxiter: int = 50,
    verbosity: Literal[0, 1, 2, 3] = 1,
) -> tuple[Float[Array, "n"], Scalar]:
    r"""Compute normalization factors for signals and their derivatives while preserving derivative consistency.

    Consider a trajectory $y(t) \in \mathbb{R}^n$ together with its derivatives
        $v(t) = \partial_t \, y(t),\; a(t) = \partial_{tt} \, y(t)$.
    Normalizing $y, v, a$ independently (e.g. to unit variance) breaks the
    derivative relationships: the derivative of the normalized trajectory
    $\partial_t \tilde y(t)$ would no longer match the normalized velocity
    $\tilde v(t)$.

    To maintain consistency, the same scaling factor $\alpha_i$ must be applied
    to $y_i, v_i, a_i$. Additionally, we allow a rescaling of time,
    $\tilde t = \tau t$, to gain more flexibility.
    With this, the rescaled signals are
    $$
        \tilde t = \tau t, \quad
        \tilde y_i = \alpha_i y_i, \quad
        \tilde v_i = \tau \alpha_i v_i, \quad
        \tilde a_i = \tau^2 \alpha_i a_i.
    $$

    This function finds the scaling factors $\alpha \in \mathbb{R}^n$ and
    $\tau \in \mathbb{R}$ that make the standard deviations of
    $\tilde y, \tilde v$ and  $\tilde a$
    as close as possible to one, while preserving derivative consistency:
    $$
        \partial_{\tilde t} \, \tilde y_i(\tilde t) = \tilde v_i(\tilde t), \quad
        \partial_{\tilde t \tilde t} \, \tilde y_i(\tilde t) = \tilde a_i(\tilde t).
    $$

    This is done via the following optimization problem:
    Given standard deviations $\sigma(y), \sigma(v), \sigma(a)$, minimize
    $$
        \sum_i \big[
            w_y (\sigma(\tilde y_i) - 1)^2 \;+\;
            w_v (\sigma(\tilde v_i) - 1)^2 \;+\;
            w_a (\sigma(\tilde a_i) - 1)^2
        \big],
    $$
    where $w_y, w_v, w_a$ control the relative importance of each term.

    Implementation Notes:
        The problem reduces to a one-dimensional optimization over $\tau$, with
        $$
            \alpha_i(\tau) = \frac{N_i(\tau)}{D_i(\tau)}, \quad
            \tau^\star = \underset{\tau}{\operatorname{argmin}}
                \sum_i -\frac{N_i(\tau)^2}{D_i(\tau)}.
        $$
        This function solves the reformulated problem using `scipy.optimize.minimize`
        with the BFGS method. Additionally:

        - Optimization is performed over $\log(\tau)$ to enforce $\tau > 0$.
        - The initial guess $\tau_0$ is computed heuristically from
        $\sigma(y), \sigma(v), \sigma(a)$.

    Args:
        std_y: Standard deviation of the signal $y(t)$.
        std_v: Standard deviation of the signal $v(t) = \partial_t y(t)$.
            Optinoal. Defaults to None.
        std_a: Standard deviation of the signal $a(t) = \partial_{tt} y(t)$.
            Optinoal. Defaults to None.
        w_y: Optimization weight for signal $y(t)$. Defaults to 1.0.
        w_v: Optimization weight for signal $v(t)$. Defaults to 1.0.
        w_a: Optimization weight for signal $a(t)$. Defaults to 1.0.
        tol: Relative error in solution for $\tau$ acceptable for convergence.
            Defaults to 1e-6.
        maxiter: Maximum number of optimization iterations to perform.
            Defaults to 50.
        verbosity: If non-zero, print messages.
            `0` : no message printing.
            `1` : non-convergence notification messages only.
            `2` : print a message on convergence too.
            `3` : print iteration results.
            Defaults to 1.

    Returns:
        Tuple containing $\alpha_i$ and $\tau$.

    """
    if TYPE_CHECKING:
        std_y = jnp.array(std_y)

    assert w_y >= 0.0, "Weight w_y must be non-negative."
    assert w_v >= 0.0, "Weight w_v must be non-negative."
    assert w_a >= 0.0, "Weight w_a must be non-negative."

    std_y = jnp.where(std_y == 0, 1, std_y)  # Avoid division by zero

    # Treat the edge case where an exact solution is available
    if (std_v is None and std_a is None) or (w_v == w_a == 0.0):
        alpha = 1 / std_y
        tau = jnp.array(1.0)  # Best arbitrary choice
        return alpha, tau

    # Initial guess
    eps = 1e-10
    alpha0 = 1.0 / (
        std_y + eps
    )  # Avoid devision by zero if one std_y is zero.

    if std_v is None or w_v == 0.0:
        std_v = jnp.zeros_like(std_y)
        tau_v = 1
    else:
        _temp = alpha0 * std_v
        tau_v = jnp.sum(_temp) / jnp.sum(_temp**2)
    if std_a is None or w_a == 0.0:
        std_a = jnp.zeros_like(std_y)
        tau_a = 1
    else:
        _temp = alpha0 * std_a
        tau_a = jnp.sqrt(jnp.sum(_temp) / jnp.sum(_temp**2))

    log_tau_init = 0.5 * float(jnp.log(tau_v * tau_a))

    # Define optimization problem
    c_y = w_y * std_y
    c_v = w_v * std_v
    c_a = w_a * std_a
    c_y_sq = c_y * std_y
    c_v_sq = c_v * std_v
    c_a_sq = c_a * std_a

    def numerator(tau: Float[Array, "1"]) -> Float[Array, "n"]:
        return c_y + c_v * tau + c_a * tau**2

    def denominator(tau: Float[Array, "1"]) -> Float[Array, "n"]:
        return c_y_sq + c_v_sq * tau**2 + c_a_sq * tau**4

    def fun(log_tau: Float[Array, "1"]):
        tau = jnp.exp(log_tau)
        loss = -(numerator(tau) ** 2) / denominator(tau)
        return jnp.sum(loss)

    # Starting bracket
    radius = 0.1
    bracket = (log_tau_init - radius, log_tau_init + radius)

    # Brent’s method in log_tau space
    res = minimize_scalar(
        fun,
        bracket=bracket,
        method="brent",
        tol=tol,
        options={"maxiter": maxiter, "xtol": tol, "disp": verbosity},
    )
    res = cast(OptimizeResult, res)

    tau = jnp.squeeze(jnp.exp(res.x))
    alpha = numerator(tau) / denominator(tau)

    return alpha, tau
