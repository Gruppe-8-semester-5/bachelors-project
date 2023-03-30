from math import sqrt
from collections.abc import Callable

import numpy as np


class Momentum:
    def __init__(self, lr: int = 0.1, beta: int = 0.9) -> None:
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