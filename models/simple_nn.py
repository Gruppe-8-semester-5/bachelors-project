import torch
import numpy as np

sigmoid = torch.nn.Sigmoid() 

def forward(X, y, weights):
    zeros = torch.zeros_like(y)
    # lg(e^0+e^x)=lg(1+e^x), but torch does not view it as e^x and then lg, but one thing.
    nll = torch.sum(torch.logaddexp(-(2 * y.double() - 1) * (X @ weights), zeros))
    return nll


def predict(w:np.ndarray, features: np.ndarray):
    """The logistic regression prediction for a single point"""
    return sigmoid(torch.from_numpy(w) @ torch.from_numpy(features)).numpy()
    # return sigmoid(w.transpose() @ features)


def negative_log_likelihood(X, y, weights):
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    weights = torch.from_numpy(weights)
    return forward(X, y, weights).numpy()

def gradient(X, y, weights):
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    weights = torch.from_numpy(weights)
    weights.requires_grad_()
    nll = forward(X, y, weights)
    nll.backward()
    return weights.grad.numpy()