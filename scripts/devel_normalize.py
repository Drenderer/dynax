# %% Testing
from typing import Literal, cast

import diffrax
import jax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Float, PRNGKeyArray
from matplotlib import pyplot as plt
from scipy.optimize import OptimizeResult, bracket, minimize_scalar

from dynax import normalization_coefficients, smooth_noise


@jax.jit
def get_interp(ts, ys):
    coeffs = diffrax.backward_hermite_coefficients(ts, ys)
    return diffrax.CubicInterpolation(ts, coeffs)


@jax.jit
def integrate(ts, y_ts):
    interp = get_interp(ts, y_ts)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: interp.evaluate(t)),
        solver=diffrax.Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=jnp.zeros_like(y_ts[0]),
        saveat=diffrax.SaveAt(ts=ts),
    )
    return sol.ys


@jax.jit
def differentiate(ts, ys):
    interp = get_interp(ts, ys)
    return jax.vmap(interp.derivative)(ts)


def generate_dummy_data(key, ts):
    keys = jr.split(key, 1)

    y_tts = (
        jax.vmap(smooth_noise, in_axes=(0, None, None, None), out_axes=-1)(
            keys, ts.size, 20, True
        )
        * 3
    )
    y_ts = integrate(ts, y_tts)
    ys = integrate(ts, y_ts)
    return ys, y_ts, y_tts


def plot(ts, ys, y_ts, y_tts):
    y_ts_ = differentiate(ts, ys)
    y_tts_ = differentiate(ts, y_ts)

    plt.plot(ts, ys, label="$y$")
    plt.plot(ts, y_ts, label=R"$\dot y$")
    plt.plot(ts, y_ts_, ls="--", label=R"$\dot \tilde y$")
    plt.plot(ts, y_tts, label=R"$\ddot y$")
    plt.plot(ts, y_tts_, ls="--", label=R"$\ddot \tilde y$")
    plt.legend()
    plt.show()


# %%


key = jr.key(0)
keys = jr.split(key, 10)
ts = jnp.linspace(0, 5, 1000)
ys, y_ts, y_tts = jax.vmap(generate_dummy_data, in_axes=(0, None))(keys, ts)

i = 0
plot(ts, ys[i], y_ts[i], y_tts[i])

# %%
std = jnp.std(ys)
std_t = jnp.std(y_ts)
std_tt = jnp.std(y_tts)

maxiter = 50
verbosity = 3
tol = 1e-8
alpha, tau = normalization_coefficients(
    std, std_t, std_tt, tol=tol, maxiter=500, verbosity=3
)
# alpha, tau = normalization_coefficients(std, std_t, None, tol=tol, maxiter=500, verbosity=3)
# alpha, tau = normalization_coefficients(std, None, None, tol=tol, maxiter=500, verbosity=3)


# %% Apply normalization:
ts_ = ts / tau
ys_ = alpha * ys
y_ts_ = tau * alpha * y_ts
y_tts_ = tau**2 * alpha * y_tts

i = 0
plot(ts, ys[i], y_ts[i], y_tts[i])

print("Standard deviations of normalized signals")
print("    y   :", jnp.std(ys_))
print("    y_t :", jnp.std(y_ts_))
print("    y_tt:", jnp.std(y_tts_))

print("Standard deviations of original  signals")
print("    y   :", jnp.std(ys))
print("    y_t :", jnp.std(y_ts))
print("    y_tt:", jnp.std(y_tts))
