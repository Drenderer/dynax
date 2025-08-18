# %% Imports
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random as jr
from jaxtyping import Array
from klax import fit

from dynax import ODESolver


# %% Define the system
class Derivative(eqx.Module):
    a_matrix: Array
    b_matrix: Array

    def __init__(self, a_matrix: Array, b_matrix: Array):
        self.a_matrix = a_matrix
        self.b_matrix = b_matrix

    def __call__(self, t, y, u):
        return -self.a_matrix @ y + self.b_matrix @ u


a_matrix = jnp.array([[0.0, 2.0], [-1.0, 0.1]])
b_matrix = jnp.array([[0.0], [1.0]])

true_system = ODESolver(Derivative(a_matrix, b_matrix))

ts = jnp.linspace(0.0, 10.0, 100)
y0 = jnp.array([1.0, 0.0])
us = jnp.sin(ts)[:, None]

ys = true_system(ts, y0, us)

# %% Plot solution
plt.plot(ts, ys)

# %% Fit the system

key = jr.key(0)
m_key, l_key = jr.split(key, 2)
a_matrix = 0.01 * jr.normal(m_key, shape=(2, 2))
b_matrix = 0.01 * jr.normal(m_key, shape=(2, 1))
learned_system = ODESolver(Derivative(a_matrix, b_matrix))


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
