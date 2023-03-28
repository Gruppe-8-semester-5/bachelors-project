import numpy as np
from models.utility import sigmoid

def predict(w:np.ndarray, features: np.ndarray):
    """The logistic regression prediction for a single point"""
    return sigmoid(w.transpose() @ features)


def negative_log_likelihood(X, y, weights):
    z = np.dot(X, weights)
    p = 1 / (1 + np.exp(-z))
    nll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return nll

def gradient(X, y, weights):
    z = np.dot(X, weights)
    p = 1 / (1 + np.exp(-z))
    grad = -np.dot(X.T, y - p)
    return grad