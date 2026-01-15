---
title: Derivative Models
---

# Derivative Models

Derivative models parameterize the right hand side of an ODE in normal form:
$$
    \dot{\mathbf{y}} = \mathbf{f}(t, \mathbf{y}[, \mathbf{u}, \text{args}])
$$
where $\mathbf{y}(t)\in\mathbb{R}^n$ is the systems state vector, $t$ is time, 
$\mathbf{u}(t)\in\mathbb{R}^m$ are external inputs (for example forces acting 
on a system), and $\text{args}$ are arbitrary parameters of the system.

They are most often combined with a [numerical solver](./integration_models.md).

## Port-Hamiltonian systems

::: dynax.ISPHS
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