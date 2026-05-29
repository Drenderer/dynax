---
title: Integration Models
---

Integration models perform numerical integration of [derivative models](./derivative_models.md) 
while supporting automatic differentiation through time.

::: dynax.ODESolver
    options:
        members:
            - __init__
            - __call__
            - get_diffrax_solution

---
For convenience, some light wrappers around the [`ODESolver`][dynax.ODESolver] are
provided:

::: dynax.AugmentedODE
    options:
        members:
            - __init__
            - __call__

::: dynax.StateSpaceSystem
    options:
        members:
            - __init__
            - __call__