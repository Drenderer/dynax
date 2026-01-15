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
            - get_augmented_trajectory
            - get_solution