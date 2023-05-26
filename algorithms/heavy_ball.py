from math import sqrt
from collections.abc import Callable

import numpy as np


class Heavy_ball:
    def __init__(self, lr: float = 0.1, beta: float = 0.9, w0 = None, step_size = None) -> None:
        """lr is learning rate, step_size is merely an added argument that can be used by the test runner class. 
           It does not relate to momentum. lr is alpha in literature"""
        if step_size is not None:
            lr = step_size
        self.lr = lr
        self.beta = beta
        self.prev_w = w0

    def step(self, w: np.ndarray, derivation: Callable[[np.ndarray], np.ndarray]):
        if self.prev_w is None:
            self.prev_w = w
        b = self.beta
        lr = self.lr
        g = derivation(w)
        momentum = w - self.prev_w
        res = w - lr * g + b * momentum
        self.prev_w = w
        return res
