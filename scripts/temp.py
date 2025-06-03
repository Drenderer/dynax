# %% Imports
from dynax import ODESolver

from klax import fit

import equinox as eqx
from jaxtyping import Array
import jax
import jax.numpy as jnp
from jax import random as jr

import matplotlib.pyplot as plt


# %% Define the system
class Derivative(eqx.Module):
    A: Array
    B: Array

    def __init__(self, A: Array, B: Array):
        self.A = A
        self.B = B

    def __call__(self, t, y, u):
        return -self.A @ y + self.B @ u


A = jnp.array([[0.0, 2.0], [-1.0, 0.1]])
B = jnp.array([[0.0], [1.0]])

true_system = ODESolver(Derivative(A, B))

ts = jnp.linspace(0.0, 10.0, 100)
y0 = jnp.array([1.0, 0.0])
us = jnp.sin(ts)[:, None]

ys = true_system(ts, y0, us)

# %% Plot solution
plt.plot(ts, ys)

# %% Fit the system

key = jr.key(0)
m_key, l_key = jr.split(key, 2)
A = 0.01 * jr.normal(m_key, shape=(2, 2))
B = 0.01 * jr.normal(m_key, shape=(2, 1))
learned_system = ODESolver(Derivative(A, B))


def loss_fn(model, data, batch_axis):
    (ts, y0, us), ys = data
    ys_pred = jax.vmap(model, in_axes=batch_axis[0])(ts, y0, us)
    return jnp.mean((ys_pred - ys) ** 2)


learned_system, hist = fit(
    learned_system,
    ((ts, y0[None, :], us), ys),
    batch_axis=((None, 0, None), None),
    loss_fn=loss_fn,
    steps=10_000,
    key=l_key,
)
hist.plot()

# %% Evaluation

ts_test = jnp.linspace(0.0, 20.0, 100)
y0_test = jnp.array([0.5, 0.0])
us_test = (jnp.exp(-0.1 * ts) * jnp.sin(ts))[:, None]

ys_true = true_system(ts_test, y0_test, us_test)
ys_pred = learned_system(ts_test, y0_test, us_test)

# %% Plot
plt.plot(ts_test, ys_true, marker="o", markevery=5, label="True System")
plt.plot(ts_test, ys_pred, linestyle="--", label="Learned System")
plt.xlabel("Time")
plt.ylabel("State")
plt.legend()
plt.show()
