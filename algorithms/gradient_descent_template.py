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
    weights = start_weights
    result = GradientDescentResult(derivation)
    
    # By hashing the input variables, we can save the result and skip the computation if it's redundant
    is_serialized, file_name = result.check_for_serialized_version(start_weights, [algorithm.__dict__[x] for x in algorithm.__dict__], type(algorithm), start_weights + algorithm.step(start_weights, derivation), derivation(start_weights), epsilon, max_iter, accuracy(weights))
    if is_serialized:
        result = result.deserialize(file_name)
        return result
    
    # Initialise
    gradient = derivation(weights)
    iteration_count = 0
    best_weight = weights
    best_accuracy = 0
    
    # Todo: Could add check to see if w or gradient changes. If not, just stop
    while not is_zero(gradient, epsilon) and (max_iter > iteration_count or max_iter == 0):
        # Add logging
        if accuracy is not None:
            # Accuracy is usally quite costly, as it computes E_in for each iteration. 
            # Remove accuracy function from arguments for speed up.
            current_accuracy = accuracy(weights)
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_weight = weights
                result.set_best_weights(best_weight)
            result.add_accuracy(current_accuracy) 
        result.add_weight(weights)

        # Perform gradient descent
        weights = algorithm.step(weights, derivation=derivation)
        iteration_count += 1
        gradient = derivation(weights)
        if iteration_count % 100 == 0:
            print(f"Gradient Descent iteration {iteration_count} @ {weights} and gradient @ {gradient}")
    
    # Save run for next time
    result.serialize(file_name)
    return result


def is_zero(w, eps): 
    return np.allclose(w, 0, 0, eps)
