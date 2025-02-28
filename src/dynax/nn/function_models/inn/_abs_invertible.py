"""
This file implements a base class for invertible layers.
"""

import equinox as eqx
from jaxtyping import Array



class InvertibleLayer(eqx.Module):

    def forward_call(self, x: Array) -> Array:
        raise NotImplementedError('The forward call is not implemented by the derived class.')
    
    def inverse_call(self, y: Array) -> Array:
        raise NotImplementedError('The forward call is not implemented by the derived class.')

    def get_forward_jacobian(self, x: Array) -> Array:
        raise NotImplementedError('The jacobian is not implemented by the derived class.')
    
    def get_inverse_jacobian(self, x: Array) -> Array:
        raise NotImplementedError('The jacobian is not implemented by the derived class.')
    
    def get_jacobian_determinant(self, x: Array) -> Array:
        raise NotImplementedError('The jacobian determinant is not implemented by the derived class.')

    def __call__(self, x: Array) -> Array:
        return self.forward_call(x)