---
title: Latent space projections
---

# Latent space projections

## Linear projections

::: dynax.PODLatentSpace
    options:
        members:
            - __init__
            - to_latent
            - from_latent

## Non-linear projections

::: dynax.CALMLayer
    options:
        members:
            - __init__
            - __call__
            - compute_kernel

### Helper classes for building CALM-based autoencoders

::: dynax.PointArgSequential
    options:
        members:
            - __init__
            - __call__

::: dynax.PointArgAE
    options:
        members:
            - encode
            - decode

::: dynax.PointArgWrapper
    options:
        members:
            - __call__

::: dynax.PointArgFunc
    options:
        members:
            - __call__