from math import sqrt
from collections.abc import Callable

import numpy as np


class Nesterov_acceleration_adaptive:
    def __init__(self, w0: np.ndarray, start_alpha: float = 0.5, overridden_beta = None) -> None:
        """Adaptive nesterov acceleration, will adjust alpha adn"""
        print("Note: Using adaptive nesterov, make sure this is intended!")
        self.alpha = start_alpha
        self._overridden_beta = overridden_beta # Overridden just means that if assigned, beta will not be adaptive
        self.beta = self.get_next_beta()
        self._prev_w = w0   # Assigning the same values gives a momentum term of 0 in the first iteration.
        self.w = w0         # Assigning the same values gives a momentum term of 0 in the first iteration.

    def get_beta(self): # beta_i
        if self.is_beta_overridden():
            return self.get_overridden_beta()
        return self.beta
    
    def is_beta_overridden(self):
        if self._overridden_beta is None:
            return False
        return True
    
    def get_overridden_beta(self):
        if self.is_beta_overridden() == False:
            raise Exception("Attempted to get overridden beta, but was not provided")
        return self._overridden_beta

    def get_next_beta(self): # beta_i+1
        if self.is_beta_overridden():
            return self.get_overridden_beta()

        alpha = self.get_alpha()
        next_alpha = self.get_next_alpha()
        return (alpha - 1) / next_alpha

    def get_alpha(self): # alpha_i
        return self.alpha
    
    def get_next_alpha(self): # alpha_i+1
        alpha = self.get_alpha()
        next_alpha = 0.5 * (alpha * np.sqrt(4 * alpha ** 2 + 1) + 1)
        if next_alpha == np.inf: # Inifite alpha found, just return previous alpha (probably close to 1.0e+259)
            return alpha
        return next_alpha
    
    def get_w(self):
        return self.w
    
    def get_prev_w(self):
        return self._prev_w

    def get_momentum_term(self):# m_i+1
        beta = self.get_beta()
        prev_w = self.get_prev_w()
        w = self.get_w()
        return beta * (w - prev_w)
 
    def step(self, w: np.ndarray, derivation: Callable[[np.ndarray], np.ndarray]):
        # Compute next w
        w = self.get_w()
        alpha = self.get_alpha()
        moment = self.get_momentum_term()
        next_w = w - alpha * (derivation(w + moment)) + moment

        # Compute adaptive parameters
        self.beta = self.get_next_beta()
        self.alpha = self.get_next_alpha()
        self._prev_w = w
        self.w = next_w
        return next_w
