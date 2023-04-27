from math import sqrt
from collections.abc import Callable

import numpy as np


class Momentum:
    def __init__(self, lr: float = 0.1, beta: float = 0.9, step_size = None) -> None:
        if step_size is not None:
            lr = step_size
        self.lr = lr
        self.beta = beta
        self.momentum = 0

    def step(self, w: np.ndarray, derivation: Callable[[np.ndarray], np.ndarray]):
        # https://en.wikipedia.org/wiki/Stochastic_gradient_descent
        # http://www.cs.utoronto.ca/~ilya/pubs/2013/1051_2.pdf
        b = self.beta
        lr = self.lr
        g = derivation(w)
        self.momentum = b * self.momentum - lr * g
        res = w + self.momentum
        return res
