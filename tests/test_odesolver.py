from dynax import ODESolver

from jax import numpy as jnp


# TODO: Add more extensive tests.
def test_odesolver():
    def func(t, x, u, c):
        return -c * x

    model = ODESolver(func)

    ts = jnp.linspace(0, 1, 100)
    y0 = jnp.array([0.5, 1.0, 2.0])
    c = 2.0

    true_sol = jnp.multiply.outer(jnp.exp(-c * ts), y0)
    pred_sol = model(ts, y0, c=c)

    assert jnp.allclose(true_sol, pred_sol)
