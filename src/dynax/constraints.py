
import jax.numpy as jnp
import equinox as eqx
import paramax
from jaxtyping import Array


class NonNegative(paramax.AbstractUnwrappable):
    """Applies non-negative constraint to each element of a weight."""
    parameter: Array = eqx.field(converter=lambda x: jnp.maximum(x, 0.))    # Ensure parameters fulfill the constraint initially

    def unwrap(self):
        return jnp.maximum(self.parameter, 0.)
    