import torch
import numpy as np

from models.utility import to_torch

sigmoid = torch.nn.Sigmoid() 
nll = torch.nn.BCEWithLogitsLoss()

def initial_params(X) -> any:
    w0 = np.random.normal(size=X.shape[1])
    return w0
def predict(w:np.ndarray, features: np.ndarray):
    """The logistic regression prediction for a single point"""
    return torch.round(sigmoid(torch.from_numpy(features @ w))).numpy()
    # return sigmoid(w.transpose() @ features)
def negative_log_likelihood(X, y, weights):
    return nll(X @ weights, y.double())
    
def gradient(X, y, weights):
    X, y, weights = to_torch(X, y, weights)
    weights.requires_grad_()
    neg = negative_log_likelihood(X, y, weights)
    neg.backward()
    return weights.grad.numpy()

def forward(X, weights):
    return sigmoid(X @ weights)
