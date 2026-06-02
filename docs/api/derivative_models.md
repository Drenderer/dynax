---
title: Derivative Models
---

# Derivative Models

Derivative models parameterize the right hand side of an ODE in normal form:
$$
    \dot{\boldsymbol{x}} = \boldsymbol{f}(t, \boldsymbol{x}[, \boldsymbol{u}; \text{args}])
$$
where $\boldsymbol{x}(t)\in\mathbb{R}^n$ is the systems state vector, $t$ is time, 
$\boldsymbol{u}(t)\in\mathbb{R}^m$ are external inputs (for example forces acting 
on a system), and $\text{args}$ is a set of arbitrary parameters of the system.

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