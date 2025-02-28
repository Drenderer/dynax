

# %% Imports

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.nn.initializers import glorot_uniform, uniform
import equinox as eqx
import paramax
import optax

from jaxtyping import Array, PRNGKeyArray
from dynax import training
from dynax.constraints import NonNegative

# %% Define a Module

class MyModel(eqx.Module):
    weight: Array|eqx.Module
    bias: Array

    def __init__(self, *, key: PRNGKeyArray):
        wkey, bkey = jr.split(key)
        x = glorot_uniform()(wkey, shape=(2,3))
        self.weight = NonNegative(x)
        self.bias = uniform(scale=1)(bkey, (2,))

    def __call__(self, x: Array) -> Array:
        return self.weight @ x + self.bias
    


def print_model_parameters(model):
    m = training.resolve_constraints(model)
    print('Bias:', m.bias, '\nWeight:\n', m.weight, '\n')

# %% Generate some training data
key = jr.key(1)
data_key, model_key = jr.split(key, 2)

w = jnp.array([[-0.2, 2., 3.], 
               [ 4. , 5., 6.]])
b = jnp.array([1., 2.])

def true_system(x):
    return w @ x + b

x = jr.uniform(data_key, shape=(100, 3))
y = jax.vmap(true_system)(x)

# %% Define and train model
model = MyModel(key=key)
print('Initially the parameters have these values:')
print_model_parameters(model)

model, h = training.fit(model, x, y, key=jr.key(0), steps=10000, optimizer=optax.adam(1e-2))

print('The value of the bias after training is:')
print_model_parameters(model)