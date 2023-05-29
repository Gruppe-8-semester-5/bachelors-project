import math
import numpy as np
import torch


def sigmoid(z):
    return 1/(1+(np.exp(-z)))

def to_torch(*vals):
    return map(torch.from_numpy, vals)

def accuracy(actual: np.ndarray, predictions: np.ndarray):
    return 1 - np.mean(actual != predictions)

def accuracy_k_encoded(labels: np.ndarray, predictions: np.ndarray):
    """Compute the accuracy of a k-encoded label and prediction, remember these are probabilities"""
    # Get the index of the maximum value for each prediction
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Compute the accuracy by comparing the predicted labels with the actual labels
    accuracy = np.mean(predicted_labels == labels)
    
    return accuracy

def make_train_and_test_sets(X: np.ndarray, y: np.ndarray, train_fraction: float):
    """
    `(x_train, y_train), (x_test, y_test) = make_train_and_test_sets(X, y, test_fraction)`
    """
    permuted_index = np.random.permutation(y.shape[0])
    X_perm = X[permuted_index]
    y_perm = y[permuted_index]
    split = np.floor(X.shape[0] * train_fraction).astype(int)
    test = X_perm[split:], y_perm[split:]
    train = X_perm[:split], y_perm[:split]
    return train, test

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
