# %% Imports
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random as jr
from jaxtyping import Array
from klax import fit

from dynax import NeuralODE, ODESolver

# %% Define a neural ODE

node_derivative = NeuralODE(
    state_size=2,
    input_size=1,
    time_dependent=False,
    state_dependent=False,
    width_sizes=[32, 32],
    key=jr.key(0),
)

node = ODESolver(node_derivative)

# %% Simulate the neural ODE
t_span = jnp.linspace(0.0, 10.0, 100)
y0 = jnp.array([1.0, 0.0])
us = 2 * t_span[:, None]
y_eval = node(t_span, y0, us)

# %% Plot the results
plt.plot(t_span, y_eval[:, 0], label="State 1")
plt.plot(t_span, y_eval[:, 1], label="State 2")
plt.xlabel("Time")
plt.ylabel("States")
plt.title("Neural ODE Simulation")
plt.legend()
plt.show()
