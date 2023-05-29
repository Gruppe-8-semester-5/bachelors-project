import math
import numpy as np
from collections.abc import Callable
from algorithms.gradient_descent_result import GradientDescentResult
from typing import Protocol
import models.softmax_regression as standard_model

from models.utility import accuracy, mini_batch_generator

class GD(Protocol):
    def step(self, w: np.ndarray, derivation: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        pass

def find_minima(algorithm: GD, X, y,
                start_weights: np.ndarray = None,
                epsilon: float = np.finfo(float).eps,
                max_iter = 1000,
                auto_stop: bool = True,
                model = standard_model,
                batch_size: int = 100,
                X_test = None,
                y_test = None,
                ):
    if start_weights is not None:
        weights = start_weights
    else:
        # weights = model.initial_params()
        weights = model.initial_params(X.shape[1], 300, 10)
    if X_test is None:
        X_test = X
        y_test = y
    # result = GradientDescentResult(derivation)
    
    # By hashing the input variables, we can save the result and skip the computation if it's redundant
    
    # Initialise
    # gradient, loss = model.gradient_and_loss(X, y, weights)
    iteration_count = 0
    best_weight = weights
    best_accuracy = 0
    
    # Todo: Could add check to see if w or gradient changes. If not, just stop
    N = math.ceil(y.shape[0] / batch_size)
    accs = []
    while iteration_count <= max_iter:
        # https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
        permuted_index = np.random.permutation(y.shape[0])
        X_perm = X[permuted_index]
        y_perm = y[permuted_index]
        for i in range(N):
            low = i * batch_size
            high = low + batch_size
            x_cur, y_cur = X_perm[low:high], y_perm[low:high]
            derivation = lambda w: model.gradient(x_cur, y_cur, w)
            weights = algorithm.step(weights, derivation=derivation)
            iteration_count += 1
        acc = accuracy(y_test, model.predict(weights, X_test))
        print(acc)
        print(iteration_count)
        accs.append(acc)
    print('DONE')
    exit()

    return result


def is_zero(w, eps): 
    res = True
    # If matrix consists of multiple weights, check all.
    if w.dtype == 'object':
        for i in range(w.shape[0]):
            res &= np.allclose(w[i], 0, 0, eps)
        return res
    return np.allclose(w, 0, 0, eps)
