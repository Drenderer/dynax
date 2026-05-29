from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.nn.initializers import Initializer, glorot_normal, normal, zeros
from jaxtyping import Array, Float, Int, PRNGKeyArray
from klax import NonTrainable
from scipy.spatial import cKDTree


class RFF(eqx.Module):
    """Random Fourier Features layer for positional encoding."""

    weight: Array  # (in_dim, num_features)

    def __init__(
        self,
        in_dim: int,
        num_features: int,
        weight_init: Initializer = normal(6.0),
        *,
        key: PRNGKeyArray,
    ):
        self.weight = weight_init(key, (in_dim, num_features))

    def __call__(
        self, x: Float[Array, "... in_dim"], *, key: PRNGKeyArray | None = None
    ) -> Float[Array, "... 2*num_features"]:
        # x: (..., in_dim)
        proj = jnp.dot(x, self.weight)
        return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)


class CALMLayer(eqx.Module):
    """Continuous and Adaptive convolution Layer.

    Implementation of "CALM-PDE" from: 10.48550/arXiv.2505.12944
    by Jan Hagnberger, Daniel Musekamp and Mathias Niepert.
    """

    in_channels: int
    out_channels: int
    query_points: Float[Array, "num_query_points num_spatial_dims"]
    receptive_size: int
    temperature: float
    modulate_kernel: bool
    candidate_indices: Int[Array, "num_query_points num_candidates"] | None
    num_candidates: int
    rff: RFF  # (..., dim) -> (..., 32)
    kernel_weight1: Float[Array, "32 32"]
    kernel_bias1: Float[Array, "num_query_points 32"]
    kernel_modulation_scale: Float[Array, "num_query_points 32"] | None
    kernel_activation: Callable[[Array], Array]
    kernel_weight2: Float[Array, "32 out_channels*in_channels"]
    kernel_bias2: Float[Array, "out_channels*in_channels"]
    bias: Float[Array, "out_channels"]
    activation: Callable[[Array], Array]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        initial_query_points: Float[Array, "num_query_points num_spatial_dims"]
        | Callable[
            [PRNGKeyArray], Float[Array, "num_query_points num_spatial_dims"]
        ],
        receptive_size: int,
        temperature: float = 1.0,
        query_points_learnable: bool = True,
        modulate_kernel: bool = True,
        initial_input_points: Float[Array, "num_input_points num_spatial_dims"]
        | None = None,
        num_candidates: int | None = None,
        weight_init: Initializer = glorot_normal(),
        bias_init: Initializer = zeros,
        rff_init: Initializer = normal(6.0),
        activation: Callable[[Array], Array] = jax.nn.gelu,
        kernel_activation: Callable[[Array], Array] = jax.nn.gelu,
        key: PRNGKeyArray,
    ):
        """Initialize CALM layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            initial_query_points: Initial query points array with shape `(num_query_points, num_spatial_dims)`.
                Alternatively, it can be a callable that takes a PRNG key and returns such an array.
            receptive_size: Number of points in the input to consider for each query point's receptive field.
                This is similar to the kernel size in discrete convolutions.
            temperature: Temperature parameter for the softmax function computing the distance weighting.
                Smaller values increase the weighting of contributions of closer input points for the
                Monte Carlo integration, while larger values tend to treat all input points equally.
                Defaults to 1.
            query_points_learnable: Whether query points are learnable. Defaults to True.
            modulate_kernel: Whether to use kernel modulation. Kernel modulation enables the use of different
                kernel functions per query point, by scaling and shifting the intermediate kernel MLP features
                with query point specific parameters. Defaults to True.
            initial_input_points: Optional initial input points array with shape `(num_input_points, num_spatial_dims)`.
                If provided, this is used to determine the indices of candidate input points per query point for
                efficient nearest neighbour computation. Instead of the inefficient brute-force nearest neighbour search
                over all input points, only the candidate input points are considered, when computing the `receptive_size`
                nearest neighbours for each query point. Candidates are the `num_candidates` closest input points to each
                query point.
                WARNING:
                    The candidate indices are computed once at initialization and remain fixed during training.
                    Therefore, if the input OR query points change significantly (This includes
                    permutations of their arrays!) during training or inference, the candidate indices may no
                    longer be valid, leading to incorrect results.
                Defaults to None.
            num_candidates: Number of input point candidates saved for the efficient nearest neighbour computation.
                If None, the `receptive_size` is used. Defaults to None.
            weight_init: Initializer for weights. Defaults to glorot_normal().
            bias_init: Initializer for biases. Defaults to zeros.
            rff_init: Initializer for the random fourier feature (RFF) weight. Defaults to normal(6.0).
            activation: Activation function for the layer output. Defaults to jax.nn.gelu.
            kernel_activation: Activation function for the kernel MLP. Defaults to jax.nn.gelu.
            key: PRNG key for initialization.

        """
        qkey, kkey, bkey = jr.split(key, 3)

        self.in_channels = in_channels
        self.out_channels = out_channels

        query_points = (
            initial_query_points(qkey)
            if callable(initial_query_points)
            else initial_query_points
        )
        num_query_points, num_spatial_dims = query_points.shape
        self.query_points = (
            query_points
            if query_points_learnable
            else NonTrainable(query_points)
        )

        assert receptive_size > 0, "receptive_size must be positive"
        self.receptive_size = receptive_size
        self.temperature = NonTrainable(temperature)
        self.modulate_kernel = modulate_kernel

        num_candidates = num_candidates or receptive_size
        self.num_candidates = num_candidates
        if initial_input_points is not None:
            if (
                not receptive_size
                <= num_candidates
                <= initial_input_points.shape[0]
            ):
                raise ValueError(
                    "num_candidates must be larger than the receptive_size and smaller than "
                    "the number of initial input points. \nThis gives the following valid "
                    f"range [{receptive_size}, {initial_input_points.shape[0]}], but got "
                    f"num_candidates={num_candidates}."
                )

            kd_tree = cKDTree(initial_input_points)
            _, candidate_indices = kd_tree.query(
                query_points, k=num_candidates
            )
            self.candidate_indices = jnp.array(
                candidate_indices[:, jnp.newaxis]
                if num_candidates == 1
                else candidate_indices,
                dtype=jnp.int32,
            )
        else:
            self.candidate_indices = None

        # Kernel related parameters
        krkey, kw1key, kb1key, kw2key, kb2key = jr.split(kkey, 5)
        self.rff = RFF(
            in_dim=num_spatial_dims,
            num_features=16,
            weight_init=rff_init,
            key=krkey,
        )
        self.kernel_weight1 = weight_init(kw1key, (32, 32))
        if modulate_kernel:
            self.kernel_modulation_scale = jnp.ones((num_query_points, 32))
            self.kernel_bias1 = bias_init(kb1key, (num_query_points, 32))
        else:
            self.kernel_modulation_scale = None
            self.kernel_bias1 = bias_init(kb1key, (32,))
        self.kernel_activation = kernel_activation
        self.kernel_weight2 = weight_init(
            kw2key, (32, out_channels * in_channels)
        )
        self.kernel_bias2 = bias_init(kb2key, (out_channels * in_channels,))
        self.bias = bias_init(bkey, (out_channels,))
        self.activation = activation

    def compute_kernel(
        self, rel_pos: Float[Array, "m k d"]
    ) -> Float[Array, "m k c o"]:
        """Compute the kernel matrix for a single query point.

        Args:
            rel_pos: Relative positions with shape `(num_query_points, num_neighbours, num_spatial_dims)`.

        Returns:
            Kernel matrix with shape `(num_query_points, num_neighbours, in_channels, out_channels)`

        """
        rff = self.rff(rel_pos)  # (m, k, 32)
        kernel = rff @ self.kernel_weight1
        if self.modulate_kernel:
            kernel *= self.kernel_modulation_scale[..., None, :]  # (m, k, 32)
        kernel += self.kernel_bias1[..., None, :]  # (m, k, 32)
        kernel = self.kernel_activation(kernel)  # (m, k, 32)
        kernel = (
            kernel @ self.kernel_weight2 + self.kernel_bias2
        )  # (m, k, c*o)
        kernel = kernel.reshape(
            *kernel.shape[:-1], self.in_channels, self.out_channels
        )
        return kernel  # (m, k, c, o)

    def __call__(
        self,
        x: Float[Array, "... n c"],
        points: Float[Array, "n d"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "... m o"], Float[Array, "m d"]]:
        """Compute the continuous convolution output.

        Args:
            x: Input features with shape `(..., num_input_points, in_channels)`.
            points: Input point coordinates with shape `(num_input_points, num_spatial_dims)`.
            key: Unused random key. Defaults to None.

        Returns:
            Output features for the query point with shape `(..., num_query_points, out_channels)`.
            Query point coordinates with shape `(num_query_points, num_spatial_dims)`.

        """
        # Check input shape! Otherwise things might silently fail, because JAX
        # won't raise when indexing outside the array.
        num_in_points = points.shape[-2]
        if x.shape[-2:] != (num_in_points, self.in_channels):
            raise ValueError(
                f"Input x has shape {x.shape}, but expected (..., {num_in_points}, {self.in_channels})"
            )

        # Number of input points: n
        # Number of query points: m
        # Number of receptive points: k
        # Number of candidates: K
        # Number of spatial dimensions: d
        # Number of input channels: c
        # Number of output channels: o

        # Compute the indices of the k closest input points for each query point in a jax jit-friendly way
        receptive_size = min(self.receptive_size, num_in_points)

        # TODO: This control flow still has parital overlap that could be eliminated.
        if self.candidate_indices is None:
            # Expensive brute force nearest neighbour search
            rel_pos = (
                points[None, :, :] - self.query_points[:, None, :]
            )  # (m, n, d)
            sq_dists = jnp.sum(rel_pos**2, axis=-1)  # (m, n)

            neg_sq_dists, indices = jax.lax.top_k(
                -sq_dists, receptive_size
            )  # (m, k)
            rel_pos = jnp.take_along_axis(
                rel_pos, indices[..., None], axis=-2
            )  # (m, k, d)

            x = jnp.take(x, indices, axis=-2)  # (..., m, k, c)
        else:
            # Only consider candidate points for nearest neighbour search
            candidate_points = jnp.take(
                points, self.candidate_indices, axis=-2
            )  # (m, K, d)

            rel_pos = (
                candidate_points - self.query_points[:, None, :]
            )  # (m, K, d)
            sq_dists = jnp.sum(rel_pos**2, axis=-1)  # (m, K)

            neg_sq_dists, indices = jax.lax.top_k(
                -sq_dists, receptive_size
            )  # (m, k)
            rel_pos = jnp.take_along_axis(
                rel_pos, indices[..., None], axis=-2
            )  # (m, k, d)

            # Map candidate indices to original input indices
            xindices = jnp.take_along_axis(
                self.candidate_indices, indices, axis=-1
            )  # (m, k)
            x = jnp.take(x, xindices, axis=-2)  # (..., m, k, c)

        # Compute distance-based kernel weightings
        norm_sq_dists = -neg_sq_dists  # (m, k)
        norm_sq_dists -= jnp.min(
            norm_sq_dists, axis=-1, keepdims=True
        )  # (m, k)
        norm_sq_dists /= (
            jnp.max(norm_sq_dists, axis=-1, keepdims=True) + 1e-8
        )  # (m, k)
        weightings = jax.nn.softmax(
            -norm_sq_dists / self.temperature, axis=-1
        )  # (m, k)

        # Kompute MLP kernel
        kernel = self.compute_kernel(rel_pos)  # (m, k, c, o)

        # Compute output
        y = jnp.einsum("...mkc,mkco,mk->...mo", x, kernel, weightings)
        y = self.activation(y + self.bias)
        return y, self.query_points
