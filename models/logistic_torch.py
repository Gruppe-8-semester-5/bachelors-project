import torch
import numpy as np

sigmoid = torch.nn.Sigmoid() 
nll = torch.nn.BCEWithLogitsLoss()

def initial_params(X) -> any:
    w0 = np.random.normal(size=X.shape[1])
    return w0
def predict(w:np.ndarray, features: np.ndarray):
    """The logistic regression prediction for a single point"""
    return torch.round(sigmoid(torch.from_numpy(features @ w))).numpy()
    # return sigmoid(w.transpose() @ features)
def gradient(X, y, weights):
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    weights = torch.from_numpy(weights)
    weights.requires_grad_()
    neg = nll(X @ weights, y.double())
    neg.backward()
    return weights.grad.numpy()

def forward(X, weights):
    return sigmoid(X @ weights)
