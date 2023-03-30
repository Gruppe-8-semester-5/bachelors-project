from math import sqrt
from collections.abc import Callable

import numpy as np


class Nesterov_acceleration:
    def __init__(self, w0: np.ndarray, alpha: int = None, beta: int = None) -> None:
        self.w = w0
        self.alpha = alpha
        self.beta = beta

    def step(self, w: np.ndarray, derivation: Callable[[np.ndarray], np.ndarray]):
        b = self.beta
        a = self.alpha
        w_k = self.w
        diff =  b * (w - w_k)
        res = w - a * derivation(w) + b * diff
        self.w = res
        return res
