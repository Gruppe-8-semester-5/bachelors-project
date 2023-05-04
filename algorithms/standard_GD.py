from math import sqrt
from collections.abc import Callable

import numpy as np


class Standard_GD:
    def __init__(self, step_size: float = 0.1, **kwargs) -> None:
        self.step_size = step_size

    def step(self, w: np.ndarray, derivation: Callable[[np.ndarray], np.ndarray]):
        step = self.step_size * derivation(w)
        w = w - step
        return w
