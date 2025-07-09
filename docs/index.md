# Dynax
Machine learning tools for dynamic systems and more build on JAX

---

!!! warning

    Dynax is still in early development and will likely see significant API changes in the near future. Likewise, the documentation is still under heavy development.

## Overview

Dynax provides functionality for implementing Neural ODE-like models. It can be used as stand-alone package but is developed together with [`klax`](https://drenderer.github.io/klax/), which provides specialized machine learning architectures, constraints, and training utilities.

## Installation

Dynax can be installed via pip 

```bash
pip install "dynax @ git+https://github.com/Drenderer/dynax.git@main"
```

or uv

```bash
uv add "dynax @ git+https://github.com/Drenderer/dynax.git@main"
```