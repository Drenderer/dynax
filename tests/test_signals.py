# TODO: Add tests for aprbs

import numpy as np
import pytest
from jax import numpy as jnp

from dynax import bandlimited_noise, smooth_noise


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


class TestBandlimitedNoise:
    def test_output_length_and_type(self, getkey):
        key = getkey()
        length = 128
        signal = bandlimited_noise(key, length)
        assert signal.shape == (length,)
        assert isinstance(signal, jnp.ndarray)

    def test_reproducibility_same_key(self, getkey):
        key = getkey()
        sig1 = bandlimited_noise(key, 64)
        sig2 = bandlimited_noise(key, 64)
        np.testing.assert_allclose(sig1, sig2, rtol=1e-7)

    def test_different_keys_produce_different_results(self, getkey):
        key1 = getkey()
        key2 = getkey()
        sig1 = bandlimited_noise(key1, 64)
        sig2 = bandlimited_noise(key2, 64)
        # Not guaranteed, but very unlikely that they're identical
        assert not jnp.allclose(sig1, sig2)

    def test_normalization_enabled(self, getkey):
        key = getkey()
        sig = bandlimited_noise(key, 256, normalize=True)
        mean = float(jnp.mean(sig))
        std = float(jnp.std(sig))
        assert abs(mean) < 1e-6
        assert abs(std - 1.0) < 1e-6

    def test_normalization_disabled(self, getkey):
        key = getkey()
        sig = bandlimited_noise(key, 256, normalize=False)
        mean = float(jnp.mean(sig))
        std = float(jnp.std(sig))
        # Without normalization, mean/std are arbitrary but finite
        assert jnp.isfinite(mean)
        assert jnp.isfinite(std)

    def test_custom_dt_changes_frequency_grid(self, getkey):
        key = getkey()
        sig_default = bandlimited_noise(key, 128, max_freq=10, dt=None)
        sig_custom = bandlimited_noise(key, 128, max_freq=10, dt=0.5)
        # Not guaranteed to be totally different, but typically will differ
        assert not jnp.allclose(sig_default, sig_custom)

    def test_large_max_freq_means_no_filtering(self, getkey):
        key = getkey()
        sig_full = bandlimited_noise(key, 128, max_freq=1e6)
        sig_default = bandlimited_noise(key, 128, max_freq=128)
        # Both should be different signals but still normalized
        assert sig_full.shape == sig_default.shape
        assert abs(float(jnp.std(sig_full)) - 1.0) < 1e-6
