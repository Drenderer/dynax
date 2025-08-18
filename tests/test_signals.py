# TODO: Add tests for aprbs

import pytest
from jax import numpy as jnp

from dynax import smooth_noise


@pytest.mark.parametrize("length", [0, 1, 200])
@pytest.mark.parametrize("start_at_zero", [False, True])
@pytest.mark.parametrize("sigma", [0.1, 10])
def test_smooth_noise(length, sigma, start_at_zero, getkey):
    signal = smooth_noise(getkey(), length, sigma, start_at_zero)
    assert signal.size == length
    if start_at_zero and length > 0:
        assert signal[0] == 0.0


@pytest.mark.parametrize(
    "length, sigma, start_at_zero",
    [
        (200, 0, False),
        (200, 0, True),
        (-1, 10, False),
        (-1, 10, True),
    ],
)
def test_smooth_noise_failing(length, sigma, start_at_zero, getkey):
    with pytest.raises(ValueError):
        smooth_noise(getkey(), length, sigma, start_at_zero)
