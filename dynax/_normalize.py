# %% Define the function
from typing import Literal, cast

from jax import numpy as jnp
from jaxtyping import Array, Float, Scalar
from scipy.optimize import OptimizeResult, bracket, minimize_scalar

# TODO: Add tests (see below)
# TODO: Add more userfriendly normalization
# TODO: Finalize docstring


def normalize_std(
    std: Float[Array, "n"],
    std_t: Float[Array, "n"] | None = None,
    std_tt: Float[Array, "n"] | None = None,
    tol: float = 1e-6,
    maxiter: int = 50,
    verbosity: Literal[0, 1, 2, 3] = 1,
) -> tuple[Float[Array, "n"], Scalar]:
    r"""Compute scaling factors for consistent signal normalization.

    Given the standard deviations of a signal $y(t) \in\mathbb{R}^n$ and (optionally)
    its first and second derivatives $\partial_t y(t), \partial_{tt} y$, computes
    the scaling factors $\alpha\in\mathbb{R}^n$ and $\tau\in\mathbb{R}$ such that the rescaled signals
    $ \tilde t * \tau = t,$
    $    \quad   \tilde y_i = \alpha_i\y_i,   \quad   \partial_{\tilde t} \tilde{y}_i = \alpha_i\partial_t y_i,   \quad   \partial_{\tilde t\tilde t} \tilde{y}_i = \alpha_i\partial_{tt} y_i $
    minimize ...

    Args:
        std: _description_
        std_t: _description_. Defaults to None.
        std_tt: _description_. Defaults to None.
        tol: _description_. Defaults to 1e-6.
        maxiter: _description_. Defaults to 50.
        verbosity: _description_. Defaults to 1.

    Raises:
        ValueError: _description_

    Returns:
        _description_
    """
    if std_tt is None:
        # Exact solution available
        alpha = 1 / std
        if std_t is None:
            tau = jnp.array(1.0)  # Best arbitrary choice
        else:
            _temp = alpha * std_t
            tau = jnp.sum(_temp) / jnp.sum(_temp**2)
        return alpha, tau

    if std_t is None:
        raise ValueError("std_t cannot be None if std_tt is provided.")

    # Numerically solve the optimization problem
    std_sq = std**2
    std_t_sq = std_t**2
    std_tt_sq = std_tt**2

    def numerator(tau: Float[Array, "1"]) -> Float[Array, "n"]:
        return std + std_t * tau + std_tt * tau**2

    def denominator(tau: Float[Array, "1"]) -> Float[Array, "n"]:
        return std_sq + std_t_sq * tau**2 + std_tt_sq * tau**4

    def fun(log_tau: Float[Array, "1"]):
        tau = jnp.exp(log_tau)
        loss = -(numerator(tau) ** 2) / denominator(tau)
        return jnp.sum(loss)

    # Initial guess
    eps = 1e-10
    alpha0 = 1.0 / (std + eps)
    tau1 = jnp.mean(1.0 / (alpha0 * (std_t + eps)))
    tau2 = jnp.sqrt(jnp.mean(1.0 / (alpha0 * (std_tt + eps))))
    log_tau_init = 0.5 * float(jnp.log(jnp.abs(tau1) * jnp.abs(tau2)))

    # Find starting bracket
    radius = 0.1
    a, b, c, fa, fb, fc, nf = bracket(
        fun, xa=log_tau_init - radius, xb=log_tau_init + radius
    )
    print(f"Took {nf} function calls to find bracket.")

    # Brent’s method in log_tau space
    res = minimize_scalar(
        fun,
        bracket=(a, b, c),
        method="brent",
        tol=tol,
        options={"maxiter": maxiter, "xtol": tol, "disp": verbosity},
    )
    res = cast(OptimizeResult, res)

    tau = jnp.squeeze(jnp.exp(res.x))
    alpha = numerator(tau) / denominator(tau)

    return alpha, tau
