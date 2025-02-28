"""
This file implements Input Convex Neural Networks (FICNNs)
The basic structure is identical to FICNNs (https://arxiv.org/abs/1609.07152) by Amos et al.
However, the weights are constrained to be positive using softplus, as described here 
(https://proceedings.neurips.cc/paper_files/paper/2019/file/0a4bbceda17a6253386bc9eb45240e25-Paper.pdf)
This does NOT make them strictly convex!
"""

import equinox as eqx

import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr

from jaxtyping import Array, PRNGKeyArray
from typing import Union, Literal, Callable


class _SICNNLayer(eqx.Module):
    weight_z: Array
    weight_y: Array
    bias:     Array

    def __init__(self,
                 y_size:int,
                 z_in_size:int,
                 z_out_size:int,
                 *,
                 key:PRNGKeyArray):

        w_z_key, w_y_key, b_key = jr.split(key, 3)
        w_initializer = jnn.initializers.glorot_uniform()
        b_initializer = jnn.initializers.normal()

        weight_z   = w_initializer(w_z_key, (z_out_size, z_in_size))
        self.weight_z = weight_z - 3    # Shift the mean below zero such that after softplus application the mean is not to large.
        self.weight_y   = w_initializer(w_y_key, (z_out_size, y_size))
        self.bias       = b_initializer(b_key, (z_out_size, ))

    def __call__(self, z, y):
        pos_weight_z = jnn.softplus(self.weight_z)
        return pos_weight_z @ z + self.weight_y @ y + self.bias
    
class SICNN(eqx.Module):
    layers: tuple[_SICNNLayer|eqx.nn.Linear, ...]
    in_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    out_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    activation: Callable

    def __init__(self,
                 in_size: Union[int, Literal["scalar"]],
                 out_size: Union[int, Literal["scalar"]],
                 width:int = 16, 
                 depth:int = 2, 
                 activation: Callable = jnn.softplus,
                 *, 
                 key:PRNGKeyArray):
        
        _in_size  = 1 if in_size =='scalar' else in_size
        _out_size = 1 if out_size=='scalar' else out_size

        keys = jr.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(eqx.nn.Linear(_in_size, _out_size, key=keys[0]))
        else:
            layers.append(eqx.nn.Linear(_in_size, width, key=keys[0]))
            for i in range(depth-1):
                layers.append(_SICNNLayer(_in_size, width, width, key=keys[i+1]))
            layers.append(_SICNNLayer(_in_size, width, _out_size, key=keys[-1]))

        self.layers = tuple(layers)
        self.activation = activation
        self.in_size = in_size
        self.out_size = out_size

    def __call__(self, y:Array):
        if self.in_size == "scalar":
            if jnp.shape(y) != ():
                raise ValueError("y must have scalar shape")
            y = jnp.broadcast_to(y, (1,))

        z = self.layers[0](y)
        z = self.activation(z)
        for layer in self.layers[1:-1]:
            z = layer(z, y)
            z = self.activation(z)
        z = self.layers[-1](z, y)

        if self.out_size == "scalar":
            assert jnp.shape(z) == (1,)
            z = jnp.squeeze(z)

        return z
