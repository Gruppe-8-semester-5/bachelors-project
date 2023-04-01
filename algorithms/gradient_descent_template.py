import numpy as np
from collections.abc import Callable
from algorithms.gradient_descent_result import GradientDescentResult
from typing import Protocol

class GD(Protocol):
    def step(self, w: np.ndarray, derivation: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        pass

def find_minima(start_weights: np.ndarray, 
                algorithm: GD,
                derivation: Callable[[np.ndarray], np.ndarray], 
                epsilon: float = np.finfo(float).eps,
                max_iter = 1000,
                accuracy: Callable[[np.ndarray], np.ndarray] = None):
    # Attempt to deserialize the results (skip this experiment if we've already done it)
    weights = start_weights
    result = GradientDescentResult(derivation)
    
    # By hashing the input variables, we can save the result and skip the computation if it's redundant
    is_serialized, serial_hash = result.check_for_serialization(start_weights, algorithm, np.sum(start_weights + algorithm.step(start_weights, derivation)), derivation(start_weights), epsilon, max_iter, accuracy(weights))
    if is_serialized:
        result = result.deserialize(serial_hash)
        return result
    
    gradient = derivation(weights)
    iteration_count = 0
    best_weight = weights
    best_accuracy = 0
    
    # Todo: Could add check to see if w or gradient changes. If not, just stop
    while not is_zero(gradient, epsilon) and (max_iter > iteration_count or max_iter == 0):
        if accuracy is not None:
            current_accuracy = accuracy(weights)
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_weight = weights
                result.set_best_weights(best_weight)
            result.add_accuracy(current_accuracy)
        result.add_weight(weights)
        weights = algorithm.step(weights, derivation=derivation)
        iteration_count += 1
        gradient = derivation(weights)
        if iteration_count % 100 == 0:
            print(f"Simple Gradient Descent iteration {iteration_count} @ {weights} and gradient @ {gradient}")    
    result.serialize(serial_hash)
    return result


def is_zero(w, eps): 
    return np.allclose(w, 0, 0, eps)