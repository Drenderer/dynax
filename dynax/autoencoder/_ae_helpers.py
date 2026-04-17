"""Helper classes to make it easier to build autoencoder architectures based on the CALM layer."""

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import equinox as eqx
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray


class PointArgFunc(Protocol):
    """Function that maps arrays of features and points to features and points."""

    def __call__(
        self, features: Array, points: Array, /
    ) -> tuple[Array, Array]: ...


class PointArgWrapper(eqx.Module):
    """Wraps a function from features to features to be compatible with the `PointArgFunc` protocol."""

    func: Callable[[Any, PRNGKeyArray | None], Any]

    def __call__(
        self,
        features: Array,
        points: Array,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Array, Array]:
        return self.func(features, key=key), points


class PointArgSequential(eqx.Module):
    """A sequence of equinox Modules adhering to the `PointArgFunc` protocol applied in sequence.

    This is a modified version of equinox' [Sequential](https://docs.kidger.site/equinox/api/nn/sequential/).
    """

    layers: tuple

    def __init__(self, layers: Sequence[PointArgFunc]):
        """Initialize the PointArgSequential.

        Args:
            layers: Sequence of `PointArgFunc`s.

        """
        self.layers = tuple(layers)

    def __call__(
        self,
        features: Array,
        points: Array,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Array, Array]:
        """Compute the sequential models output.

        Args:
            features: Features passed to the first layer.
            points: Points passed to the first layer.
            key: If provided, then it is split by the number of layers, and a subkey
                passed to each layer.

        Returns:
            Output of the last layer, which is a tuple of features and points.

        """
        if key is None:
            keys = [None] * len(self.layers)
        else:
            keys = jr.split(key, len(self.layers))
        for layer, key in zip(self.layers, keys):
            features, points = layer(features, points, key=key)
        return features, points

    def __getitem__(self, i: int | slice) -> PointArgFunc:
        """Get a layer or a sub-sequence of layers."""
        if isinstance(i, int):
            return self.layers[i]
        elif isinstance(i, slice):
            return PointArgSequential(self.layers[i])
        else:
            raise TypeError(f"Indexing with type {type(i)} is not supported")

    def __iter__(self):
        """Iterate over the layers."""
        yield from self.layers

    def __len__(self):
        """Get the number of layers."""
        return len(self.layers)


class PointArgAE(eqx.Module):
    """Autoencoder with PointArgFunc encoder and decoder."""

    encoder: PointArgFunc
    decoder: PointArgFunc

    def encode(
        self,
        features: Array,
        points: Array,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Array, Array]:
        return self.encoder(features, points)

    def decode(
        self,
        features: Array,
        points: Array,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Array, Array]:
        return self.decoder(features, points)
