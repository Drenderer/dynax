"""
This file implements a sequential model for the invertible layers.
"""

from jaxtyping import Array
from ._abs_invertible import InvertibleLayer



class INNSequential(InvertibleLayer):
    layers: list

    def __init__(self, layers: list):
        self.layers = layers

    def forward_call(self, x: Array) -> Array:
        for l in self.layers:
            x = l.forward_call(x)
        return x
    
    def inverse_call(self, y: Array) -> Array:
        for l in self.layers[::-1]:
            y = l.inverse_call(y)
        return y