"""Collection of functions for generating synthetic signals."""

# TODO: Add tests for aprbs!
# TODO: Implement (infinetly differentiable) perlin noise.

import jax.numpy as jnp
import jax.random as jr
from jax.scipy.signal import convolve
from jaxtyping import Array, PRNGKeyArray


def aprbs(
    key: PRNGKeyArray,
    length: int,
    num_jumps: int,
    initial_value: float | None = None,
) -> Array:
    """Generate an amplitude-modulated pseudo-random binary sequence (APRBS).

    The output sequence contains numbers from [0, 1).

    Args:
        key: JAX PRNGKey.
        length: Number of samples in the sequence.
        num_jumps: Number of jumps in the sequence.
        initial_value: The inital value of the sequence.
            If None, it is chosen randomly.

    Raises:
        ValueError: If the number of jumps exceeds the number
            of possible jumping points (`length`-2).

    Returns:
        Array with shape `(length,)` describing the APRBS.

    """
    keys = jr.split(key, 3)
    ts = jnp.arange(length)

    if num_jumps > length - 2:
        raise ValueError("Number of jumps must be smaller than the length-2!")

    indices = jnp.sort(
        jr.choice(keys[1], ts[1:-1], shape=(num_jumps,), replace=False)
    )
    if initial_value is None:
        values = jr.uniform(keys[2], shape=(num_jumps + 1,))
    else:
        values = jr.uniform(keys[2], shape=(num_jumps,))
        values = jnp.concatenate([jnp.array([initial_value]), values])

    counts = jnp.diff(
        jnp.concatenate([jnp.array([0]), indices, jnp.array([length])])
    )

    return jnp.repeat(values, counts, total_repeat_length=length)


def smooth_noise(
    key: PRNGKeyArray,
    length: int,
    sigma: float,
    start_at_zero: bool = False,
) -> Array:
    """Generate smooth noise.

    Produces a smoothly varying signal with unit variance by
    convonvling a Gaussian kernel with white noise.

    Args:
        key: JAX PRNGKey.
        length: Number of samples in the sequence.
        sigma: Standard deviation of the Gaussian kernel.
            Larger values produce smoother outputs.
        start_at_zero: If true starts the sequence at zero.
            Otherwise it starts with a random normaly distributed value.
            Defaults to False.

    Returns:
        Array with shape `(length,)` describing the smooth noise.

    """
    if sigma <= 0.0:
        raise ValueError("Kernel sigma must be positive!")
    if length < 0:
        raise ValueError("Signal lenth must be non-negative")
    elif length == 0:
        return jnp.empty((0,))

    kernel_radius = int(3 * sigma)  # Cutoff kernel at 3 sigma
    x = jnp.arange(-kernel_radius, kernel_radius + 1)
    kernel = jnp.exp(-0.5 * (x / sigma) ** 2)
    kernel /= jnp.sum(kernel)

    # Check if the kernel edges are close to machine precision
    print(kernel[0])
    eps = jnp.finfo(kernel.dtype).eps
    if jnp.isclose(kernel[0], eps):
        print("Close!")

    white = jr.normal(key, (length,))
    if start_at_zero:
        white = white.at[: kernel_radius + 1].set(0.0)
    white /= jnp.sqrt(jnp.sum(kernel**2))  # normalize

    return convolve(white, kernel, mode="same")


def bandlimited_noise(
    key: PRNGKeyArray,
    length: int,
    max_freq: float = 20,
    dt: float | None = None,
    normalize: bool = True,
) -> Array:
    """Generate 1D band-limited random noise using Fourier synthesis.

    Note:
        The signal is periodic due to the Fourier-based construction.

    Args:
        key: JAX PRNGKey.
        length: Number of points
        max_freq: Cutoff frequency (max Fourier mode kept)
        dt: Time increment between samples. If none provided, uses `dt=1/length`.
            Defaults to None.
        normalize: If true, normalizes the signal to zero mean and unit variance.

    Returns:
        Smooth noise of length N

    """
    # --- step 1: generate Gaussian random complex coefficients ---
    # rfft is used since we want real-valued output
    key_r, key_i = jr.split(key)
    coeff_shape = (length // 2 + 1,)
    coeffs = jr.normal(key_r, coeff_shape) + 1j * jr.normal(key_i, coeff_shape)
    coeffs *= length

    # --- step 2: apply frequency cutoff ---
    if dt is None:
        dt = 1 / length if length != 0 else 1
    freqs = jnp.fft.rfftfreq(length, d=dt)
    coeffs = jnp.where(freqs <= max_freq, coeffs, 0)

    # --- step 3: inverse FFT to get smooth signal ---
    signal = jnp.fft.irfft(coeffs, n=length)

    # --- step 4: normalize to unit std (optional) ---
    if normalize:
        signal = (signal - jnp.mean(signal)) / jnp.std(signal)

    return signal
