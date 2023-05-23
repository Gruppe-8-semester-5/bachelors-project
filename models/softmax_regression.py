import torch
import numpy as np
from models.utility import to_torch

softmax = torch.nn.Softmax(dim = 1) 
log_softmax = torch.nn.LogSoftmax(dim = 1)
nll_loss = torch.nn.NLLLoss()

def initial_params(X, y):
    W = np.random.normal(size=(X.shape[1], np.amax(y) + 1))
    return W

def predict(w:np.ndarray, features: np.ndarray):
    """The logistic regression prediction for a single point"""
    return torch.argmax(softmax(*to_torch(features @ w)), dim=1).numpy()

def loss(X, y, W):
    pred = log_softmax(X @ W)
    return nll_loss(pred, y.long())

def gradient(X, y, weights):
    X, y, W = to_torch(X, y, weights)
    W.requires_grad_()
    nll = loss(X, y, W)
    nll.backward()
    return W.grad.numpy()
