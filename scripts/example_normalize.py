# %% Testing
from dataclasses import dataclass, fields

import jax
import numpy as np
from diffrax import CubicInterpolation, backward_hermite_coefficients
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, ArrayLike
from matplotlib import pyplot as plt

from dynax import bandlimited_noise, normalization_coefficients


@dataclass
class Trajectory:
    ts: Array
    ys: Array
    y_ts: Array
    y_tts: Array

    def plot(self):
        plt.plot(self.ts, self.ys, label="$y$")
        plt.plot(self.ts, self.y_ts, label=R"$\dot y$")
        plt.plot(self.ts, self.y_tts, label=R"$\ddot y$")
        plt.legend()
        plt.show()


@jax.jit
def differentiate(ts, ys):
    coeffs = backward_hermite_coefficients(ts, ys)
    interp = CubicInterpolation(ts, coeffs)
    return jax.vmap(interp.derivative)(ts)


# %% Generate dummy data

key = jr.key(0)

dt = 0.01
n = 4
max_freq = 0.7

ts = jnp.arange(0, 10, dt)

amplitudes = jr.uniform(key, (n,))
ys = amplitudes * jax.vmap(
    lambda k: bandlimited_noise(k, ts.size, max_freq=max_freq, dt=dt),
    out_axes=-1,
)(jr.split(key, n))
y_ts = differentiate(ts, ys)
y_tts = differentiate(ts, y_ts)

traj = Trajectory(ts, ys, y_ts, y_tts)
traj.plot()

# %% Compute component wise normalization
std = jnp.std(traj.ys, axis=0)
std_t = jnp.std(traj.y_ts, axis=0)
std_tt = jnp.std(traj.y_tts, axis=0)

maxiter = 50
verbosity = 0
tol = 1e-8
alpha, tau = normalization_coefficients(
    std, std_t, std_tt, tol=tol, maxiter=maxiter, verbosity=verbosity
)

ts_ = ts / tau
ys_ = alpha * ys
y_ts_ = tau * alpha * y_ts
y_tts_ = tau**2 * alpha * y_tts

traj_norm = Trajectory(ts_, ys_, y_ts_, y_tts_)
traj_norm.plot()

print("Standard deviations of normalized signals")
print("    y   :", jnp.std(traj_norm.ys, axis=0))
print("    y_t :", jnp.std(traj_norm.y_ts, axis=0))
print("    y_tt:", jnp.std(traj_norm.y_tts, axis=0))

print("Standard deviations of original  signals")
print("    y   :", jnp.std(traj.ys, axis=0))
print("    y_t :", jnp.std(traj.y_ts, axis=0))
print("    y_tt:", jnp.std(traj.y_tts, axis=0))
