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
                gradient_and_loss: Callable[[np.ndarray], (np.ndarray, float)], 
                epsilon: float = np.finfo(float).eps,
                max_iter = 1000,
                auto_stop: bool = True,
                accuracy: Callable[[np.ndarray], np.ndarray] = None,
                serialize = True,
                accuracy_compute_interval = 1): # Accuracy_compute_interval denotes how often accuracy should be recomputed (not applicable for the first 100 iterations)
    weights = start_weights
    result = GradientDescentResult(derivation)
    
    # By hashing the input variables, we can save the result and skip the computation if it's redundant
    is_serialized, file_name = result.check_for_serialized_version(start_weights, vars(algorithm), [algorithm.__dict__[x] for x in algorithm.__dict__], type(algorithm), start_weights + algorithm.step(start_weights, derivation), derivation(start_weights), epsilon, max_iter, accuracy(weights))
    if is_serialized:
        result = result.deserialize(file_name)
        return result
    
    # Initialise
    gradient, loss = gradient_and_loss(weights)
    iteration_count = 0
    best_weight = weights
    best_accuracy = 0
    current_accuracy = 0
    while True:
        # Add logging
        if accuracy is not None:
            # Accuracy is usally quite costly, as it computes E_in for each iteration. 
            # Remove accuracy function from arguments for speed up.
            if (iteration_count > 100 and iteration_count % accuracy_compute_interval == 0) or iteration_count == max_iter:
                current_accuracy = accuracy(weights)
            else:
                current_accuracy = accuracy(weights)

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_weight = weights
                result.set_most_accurate_weights(best_weight)
            result.add_accuracy(current_accuracy) 
        result.add_weight(weights)
        result.add_loss(loss)
        result.add_grad_norm(safe_norm(gradient))

        # Perform gradient descent
        weights = algorithm.step(weights, derivation=derivation)
        iteration_count += 1
        gradient, loss = gradient_and_loss(weights)
        if iteration_count % 100 == 0:
            print(f"Gradient Descent iteration {iteration_count} @ {weights} and gradient @ {gradient}")
        if (auto_stop and is_zero(gradient, epsilon)) or (max_iter <= iteration_count):
            break
        
    # Save run for next time
    if serialize:
        result.serialize(file_name)
    return result

def safe_norm(w): 
    if w.dtype == 'object':
        sum = 0
        for i in range(w.shape[0]):
            sum += np.sum(w[i] ** 2)
        return np.sqrt(sum)
    return np.sqrt(np.sum(w ** 2))

def is_zero(w, eps): 
    res = True
    # If matrix consists of multiple weights, check all.
    if w.dtype == 'object':
        for i in range(w.shape[0]):
            res &= np.allclose(w[i], 0, 0, eps)
        return res
    return np.allclose(w, 0, 0, eps)
