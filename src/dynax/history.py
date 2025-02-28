"""
WIP WIP WIP
"""

import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt

class History:
    metrics: dict
    steps: NDArray
    training_time: float    # Time it took the trainig in seconds
    
    def __init__(self, metrics: dict[str, list[float]|NDArray], steps: list[float]|NDArray, training_time: float):
        self.metrics = {k: np.array(v) for k,v in metrics.values()}
        self.steps = np.array(steps)
        self.training_time = training_time

    def plot(self) -> None:
        for metric, values in self.metrics.items():
            plt.semilogy(self.steps, values, label=metric)


    