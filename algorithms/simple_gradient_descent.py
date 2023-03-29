import numpy as np
from collections.abc import Callable
from algorithms.gradient_descent_result import GradientDescentResult

def find_minima(w: np.ndarray, 
                step_size: float, 
                derivation: Callable[[np.ndarray], np.ndarray], 
                epsilon: float = np.finfo(float).eps,
                max_iter = 1000,
                accuracy: Callable[[np.ndarray], np.ndarray] = None):
    result = GradientDescentResult(derivation)
    gradient = derivation(w)
    iteration_count = 0

    best_weight = w
    best_accuracy = 0
    
    # Todo: Could add check to see if w or gradient changes. If not, just stop
    while not is_zero(gradient, epsilon) and (max_iter > iteration_count or max_iter == 0):
        current_accuracy = accuracy(w)
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_weight = w
        result.add_accuracy_point(current_accuracy)
        result.add_point(w)
        step = step_size * gradient
        w = w - step
        iteration_count += 1
        gradient = derivation(w)
        if iteration_count % 100 == 0:
            print(f"Simple Gradient Descent iteration {iteration_count} @ {w} and gradient @ {gradient}")    
    result.set_best_weights(best_weight)
    return result


def is_zero(w, eps): 
    return np.allclose(w, 0, 0, eps)