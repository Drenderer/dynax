# %% Imports
import equinox as eqx
import jax
import jax.numpy as jnp
import klax
import matplotlib.pyplot as plt
from jax import random as jr
from jaxtyping import Array

from dynax import (
    ISOPHS,
    ConvexNormalization,
    MatrixWrapper,
    NoArgsPotential,
    ODESolver,
)

# %%

f = klax.nn.PICNN(2, 1, "scalar", [(8, 4), (8, 4)], key=jr.key(0))
H = ConvexNormalization(
    f, state_size=2, key=jr.key(0), epsilon=jnp.array(1e-6)
)
J = MatrixWrapper(
    klax.nn.ConstantSkewSymmetricMatrix(shape=(2, 2), key=jr.key(0)),
    state_dependent=False,
    parameter_dependent=False,
)
R = MatrixWrapper(
    klax.nn.SPDMatrix(3, (2, 2), [8], key=jr.key(0)),
    state_dependent=True,
    parameter_dependent=True,
)
phs = ISOPHS(
    hamiltonian=H,
    structure_matrix=J,
    dissipation_matrix=R,
)
model = ODESolver(phs)
model = klax.finalize(model)

xs = model(
    ts=jnp.linspace(0.0, 100.0, 1000),
    x0=jnp.array([1.0, 0.0]),
    args=jnp.array([-1.0]),
)

plt.plot(xs[:, 0], xs[:, 1])

# %%
