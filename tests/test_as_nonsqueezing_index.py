import numpy as np
import pytest
from jax import numpy as jnp

from dynax._datahandling import _as_nonsqueezing_index


@pytest.mark.parametrize(
    "index, expected_shape",
    [
        # Single integer index (should add axis)
        (0, (1, 4, 5)),
        # Single slice (should not change shape)
        (slice(1, 3), (2, 4, 5)),
        # Single boolean array
        (np.array([True, False, True]), (2, 4, 5)),
        # Single 0-dim array (should add axis)
        (np.array(0), (1, 4, 5)),
        (jnp.array(0), (1, 4, 5)),
        # Tuple: int, slice, int
        ((1, slice(2, 4), 0), (1, 2, 1)),
        # Tuple: 0-dim array, slice, int
        ((np.array(0), slice(2, 4), 0), (1, 2, 1)),
        # Tuple: int, int, int
        ((1, 2, 0), (1, 1, 1)),
        # Tuple: slice, slice, slice
        ((slice(0, 2), slice(1, 3), slice(0, 2)), (2, 2, 2)),
        # Tuple: int, boolean array, int
        ((0, np.array([True, False, True, False]), 0), (1, 2, 1)),
        # Tuple: 0-dim jax array, numpy array, int
        ((jnp.array(0), np.array([1, 2]), 0), (1, 2, 1)),
        # Empty tuple:
        ((), (3, 4, 5)),
    ],
)
def test_as_nonsqueezing_index(index, expected_shape):
    arr = np.zeros((3, 4, 5))
    idx = _as_nonsqueezing_index(index)
    result = arr[idx]
    assert result.shape == expected_shape
