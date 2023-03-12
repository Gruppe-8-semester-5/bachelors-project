import numpy as np
from collections.abc import Callable
from algorithms.gradient_descent_result import GradientDescentResult

def find_minima(w: np.ndarray, 
                step_size: float, 
                function: Callable[[np.ndarray], np.ndarray],
                derivation: Callable[[np.ndarray], np.ndarray], 
                epsilon: float = np.finfo(float).eps,
                max_iter = 1000):
    result = GradientDescentResult(function, derivation)
    gradient = derivation(w)
    iteration_count = 0
    while not is_zero(gradient, epsilon) and (max_iter > iteration_count or max_iter == 0):
        result.add_point(w)
        step = step_size * gradient
        w = w - step
        iteration_count += 1
        gradient = derivation(w)
    return result


def is_zero(w, eps): 
    return np.allclose(w, 0, 0, eps)