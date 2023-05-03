import math
import numpy as np


def sigmoid(z):
    return 1/(1+(np.exp(-z)))


def accuracy(actual: np.ndarray, predictions: np.ndarray):
    return 1 - np.mean(actual != predictions)
    # return 1 - (np.sum(np.abs(actual - predictions)) / len(predictions))

def mini_batch_generator(X, y, batch_size: int):
    N = math.ceil(y.shape[0] / batch_size)
    while True:
        # https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
        permuted_index = np.random.permutation(y.shape[0])
        X_perm = X[permuted_index]
        y_perm = y[permuted_index]
        for i in range(N):
            low = i * batch_size
            high = low + batch_size
            yield X_perm[low:high], y_perm[low:high]

def make_mini_batch_gradient(X, y, batch_size: int, gradient_func):
    generator = mini_batch_generator(X,y,batch_size)
    return lambda w: gradient_func(*next(generator), w)
