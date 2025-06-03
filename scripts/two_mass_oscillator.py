# %% Imports
from dynax import ODESolver


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
    J: Array
    R: Array
    Q: Array

    def __init__(self, m1, m2, c1, c2, d1, d2):
        zeros = jnp.zeros((2, 2))

        # Structure matrix
        mass = jnp.array([[m1, 0], [0, m2]])
        mass_inv = jnp.linalg.inv(mass)
        J = jnp.block([[zeros, mass_inv], [-mass_inv, zeros]])
        self.J = J

        # Resistive matrix
        diss = jnp.array(
            [
                [(d1 + d2) / (m1 * m1), -d2 / (m1 * m2)],
                [-d2 / (m1 * m2), d2 / (m1 * m2)],
            ]
        )
        R = jnp.block([[zeros, zeros], [zeros, diss]])
        self.R = R

        # Hamililtonian quadratic form H=xQx
        Q = jnp.array(
            [[c1 + c2, -c2, 0, 0], [-c2, c2, 0, 0], [0, 0, m1, 0], [0, 0, 0, m2]]
        )
        self.Q = Q

        self.A = (J - R) @ Q

        # Input matrix
        self.B = jnp.array([0, 0, 0, 1 / m2])[:, None]

    def __call__(self, t, y, u):
        return self.A @ y + self.B @ u


deriv = Derivative(m1=1, m2=2, c1=5, c2=2, d1=0.1, d2=0.1)
true_system = ODESolver(deriv)

key = jr.key(0)
data_key, model_key, loader_key = jr.split(key, 3)

num_ts = 100
num_trajs = 2
ts = jnp.linspace(0.0, 10.0, num_ts)
y0 = jr.uniform(data_key, (num_trajs, 4))
us = jr.uniform(data_key, (num_trajs, num_ts, 1), minval=-1.0, maxval=1.0)
ys = jax.vmap(true_system, in_axes=(None, 0, 0))(ts, y0, us)

# %% Define and train sPHNN
pass

# %% Plot the prediction
fix, ax = plt.subplots()

ax.plot(ts, us[0, :, 0], label="$u$")
ax.plot(ts, ys[0, :, :2], label=["$q_1$", "$q_2$"])
ax.legend()
plt.show()
