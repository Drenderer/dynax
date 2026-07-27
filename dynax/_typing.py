from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Array, ArrayLike, Real

if TYPE_CHECKING:
    RealScalarLike = int | float | Array | np.ndarray
else:
    RealScalarLike = Real[ArrayLike, ""]
