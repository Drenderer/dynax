

# %% Imports
from typing import TypeVar

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.nn.initializers import glorot_uniform, uniform
import equinox as eqx
import paramax
import optax

from jaxtyping import Array, PRNGKeyArray

T = TypeVar("T")

# %% Define a Module

class NonNegative(paramax.AbstractUnwrappable):
    """Applies non-negative constraint to each element of a weight."""
    parameter: Array = eqx.field(converter=lambda x: jnp.maximum(x, 0.))    # Ensure parameters is positive in the first place

    def unwrap(self):
        return jnp.maximum(self.parameter, 0.)


class Trainable(paramax.AbstractUnwrappable[T]):
    """Signals that the parameter shall be trained and nothing more."""

    tree: T

    def unwrap(self) -> T:
        """Returns the wrapped tree without changes."""
        return self.tree


class MyModel(eqx.Module):
    weight: Array|paramax.AbstractUnwrappable
    bias: Array|paramax.AbstractUnwrappable

    def __init__(self, *, key: PRNGKeyArray):
        wkey, bkey = jr.split(key)
        self.weight = NonNegative(glorot_uniform()(wkey, shape=(2,3)))
        self.bias = paramax.NonTrainable(uniform(scale=1)(bkey, (2,)))

    def __call__(self, x: Array) -> Array:
        return self.weight @ x + self.bias

def train(model, x, y, steps=1000):
    # Does not use batches, trains on entire dataset

    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        model = paramax.unwrap(model)
        y_pred = jax.vmap(model)(x)
        return jnp.mean(jnp.square(y - y_pred))

    # Important for efficiency whenever you use JAX: wrap everything into a single JIT
    # region.
    @eqx.filter_jit
    def make_step(model, x, y, opt_state):
        loss, grads = compute_loss(model, x, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    optim = optax.adam(1e-2)
    opt_state = optim.init(model)
    loss = None
    for step in range(steps):
        loss, model, opt_state = make_step(model, x, y, opt_state)
        loss = loss.item()
    print(f"\nTraining finished, final loss={loss}")

    return model

def print_model_and_parameters(model):
    m = paramax.unwrap(model)
    print("Model: ", model)
    print("Unwrapped model: ", m)
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
print('\n\n\tInitially the parameters have these values:')
print("\t-------------------------------------------")
print_model_and_parameters(model)

model = train(model, x, y)

print('\n\n\tThe value of the bias after training is:')
print("\t-------------------------------------------")
print_model_and_parameters(model)

# Unfreeze bias
print('\n\n\tThe bias remained unchanged because it is frozen.\n\tNow lets unfreeze and train again.')
print("\t-------------------------------------------")
# model = eqx.tree_at(lambda x: x.bias, model, paramax.unwrap(model).bias)
def make_trainable(leaf):
    if isinstance(leaf, paramax.NonTrainable):
        return Trainable(leaf.tree)
    else:
        return leaf
model = jax.tree.map(f=make_trainable, tree=model, is_leaf=lambda x: isinstance(x, paramax.NonTrainable))

model = train(model, x, y)

print('\n\n\tThe value of the bias after the second training is:')
print("\t-------------------------------------------")
print_model_and_parameters(model)
