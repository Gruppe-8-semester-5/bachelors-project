from math import sqrt
from collections.abc import Callable

import numpy as np


class Nesterov_acceleration:
    def __init__(self, w0: np.ndarray, L: float = 0.01, mu: float = None, alpha: float = None, beta: float = None, step_size = None) -> None:
        self.w0 = w0
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.w_prev = w0
        self.w_cur = w0
        self.L = L

        if step_size != None:
            alpha = step_size
        
        if self.mu == 0 or self.mu is None:
            self.t_k = 0
            self.mu = None
        else:
            self.t_k = None
            self.beta = self.beta_mu_strong(self.mu, self.L)
        # self.mu = mu if mu != 0 else None
        if self.alpha is None:
            self.alpha = 1 / L
        else:
            self.alpha = alpha 
        if self.beta is not None:
            self.beta = beta

    def beta_mu_strong(self, L:float, mu: float):
        return (sqrt(L)-sqrt(mu)) / (sqrt(L) + sqrt(mu))

    def get_beta(self):
        if self.beta is not None:
            return self.beta
        t_k = self.t_k
        self.t_k = 1/2 * (1+sqrt(1 + 4 * self.t_k ** 2))
        return (t_k - 1) / self.t_k

    def get_alpha(self):
        return self.alpha

    def step(self, w: np.ndarray, derivation: Callable[[np.ndarray], np.ndarray]):
        b = self.get_beta()
        a = self.get_alpha()
        w_k = self.w_cur
        w_k_1 = self.w_prev
        diff =  b * (w_k - w_k_1)
        res = w - a * derivation(self.w0 + diff) + diff
        self.w_prev = self.w_cur
        self.w_cur = w
        return res
