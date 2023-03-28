import numpy as np


def sigmoid(z):
    return 1/(1+(np.exp(-z)))


def accuracy(actual: np.ndarray, predictions: np.ndarray):
    return 1 - (np.sum(np.abs(actual - predictions)) / len(predictions))