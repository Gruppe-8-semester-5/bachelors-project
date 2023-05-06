import numpy as np
import torch
from models.utility import sigmoid

def initial_params(X) -> any:
    w0 = np.random.rand(X.shape[1])
    return w0

def predict_with_softmax(features: np.ndarray, w:np.ndarray):
    """Adds a soft-max step to prediction todo: 
       fix numerical issues with softmax"""
    return torch.argmax(soft_max(features, w), dim=1).numpy()

def predict(w:np.ndarray, features: np.ndarray) -> int:
    """The logistic regression prediction for a single point. Gives outputs of {-1, 1}"""
    return np.sign(sigmoid(features @ w) * 2 - 1)


def negative_log_likelihood(X, y, weights):
    z = np.dot(X, weights)
    p = 1 / (1 + np.exp(-z))
    nll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return nll

def gradient(X, y, weights):
    """The gradient for negative log likelihood
       Expects labels to be 0 or 1"""
    # Convert if labels are -1 and 1
    y = [0. if l <= 0 else 1. for l in y]
    z = np.dot(X, weights)
    p = 1 / (1 + np.exp(-z))
    grad = -np.dot(X.T, y - p)
    return grad

def gradient_k(X, y, weights) -> np.ndarray:
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    weights = torch.from_numpy(weights)
    weights.requires_grad_()
    nll = torch.nn.NLLLoss()
    nll_tensor = nll(soft_max(X, weights), y.long())
    nll_tensor.backward()
    return weights.grad.numpy() 

def soft_max(X, w) -> torch.Tensor:
    softmax = torch.nn.Softmax(1)
    if type(X) == np.ndarray:
        X = torch.from_numpy(X)
    if type(w) == np.ndarray:
        w = torch.from_numpy(w)
    return softmax(X @ w)


def gradient_regularized(X, y, weights, lda = 0.5):
    z = np.dot(X, weights)
    p = 1 / (1 + np.exp(-z))
    grad = -np.dot(X.T, y - p)
    return grad + (lda * 2 * weights)