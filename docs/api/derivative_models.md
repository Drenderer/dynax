---
title: Derivative Models
---

# Derivative Models

Derivative models parameterize the right hand side of an ODE in normal form:
$$
    \dot{\boldsymbol{y}} = \boldsymbol{f}(t, \boldsymbol{y}[, \boldsymbol{u}; \mu])
$$
where $\boldsymbol{y}(t)\in\mathbb{R}^n$ is the systems state vector, $t$ is time, 
$\boldsymbol{u}(t)\in\mathbb{R}^m$ are external inputs (for example forces acting 
on a system), and $\mu$ is a set of arbitrary parameters of the system.

They are most often combined with a [numerical solver](./integration_models.md).

## Port-Hamiltonian systems

::: dynax.ISOPHS
    options:
        members:
            - __init__
            - __call__

::: dynax.MatrixWrapper
    options:
        members:
            - __init__
            - __call__


## Physics-agnostic systems

::: dynax.NeuralODE
    options:
        members:
            - __init__
            - __call__