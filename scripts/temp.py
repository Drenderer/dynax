# %% Imports
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random as jr
from jaxtyping import Array
from klax import fit

from dynax import CALMLayer, PointArgFunc
