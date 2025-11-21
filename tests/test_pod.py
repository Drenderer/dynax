import numpy as np
import pytest
from jax import numpy as jnp
from jax import random as jr

from dynax import PODLatentSpace, get_svd


@pytest.mark.parametrize("shape", [(10, 50, 100), (10, 50, 1000)])
def test_get_svd(shape, getkey):
    n_batch, n_time, n_space = shape
    snapshots = jr.uniform(getkey(), shape)

    lsv, sv, rsv = get_svd(snapshots)

    assert jnp.allclose(
        snapshots, lsv @ jnp.diag(sv) @ rsv, rtol=1e-4, atol=1e-4
    )
    x = min(n_batch * n_time, n_space)
    assert lsv.shape == (n_batch, n_time, x)
    assert sv.shape == (x,)
    assert rsv.shape == (x, n_space)


class TestPODLatentSpace:
    @pytest.mark.parametrize("fit_shape", [(10, 50, 100), (10, 50, 1000)])
    @pytest.mark.parametrize(
        "eval_first_dims", [(), (10,), (50, 2), (1, 2, 3)]
    )
    @pytest.mark.parametrize("num_modes", [0, 1, 10, 10_000])
    def test_basic_usage(self, fit_shape, eval_first_dims, num_modes, getkey):
        snapshots = jr.uniform(getkey(), fit_shape)
        pod = PODLatentSpace(
            snapshots, num_modes=num_modes, shift=None, scaling=None
        )

        fom_data = jr.uniform(getkey(), eval_first_dims + fit_shape[-1:])
        l = pod.to_latent(fom_data)
        assert l.shape == eval_first_dims + (pod.num_modes,)
        reconstructed_data = pod.from_latent(l)
        assert reconstructed_data.shape == eval_first_dims + fit_shape[-1:]

    @pytest.mark.parametrize("fit_shape", [(10, 50, 100), (10, 50, 1000)])
    def test_full_reconstruction(self, fit_shape, getkey):
        snapshots = jr.uniform(getkey(), fit_shape)
        x = min(fit_shape[0] * fit_shape[1], fit_shape[2])
        pod = PODLatentSpace(snapshots, num_modes=x, shift=None, scaling=None)

        latent = pod.to_latent(snapshots)
        recons = pod.from_latent(latent)
        assert jnp.allclose(snapshots, recons, rtol=1e-4, atol=1e-4)
