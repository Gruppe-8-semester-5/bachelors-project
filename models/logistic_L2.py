import torch
import models.logistic_torch as original
from models.utility import to_torch

sigmoid = torch.nn.Sigmoid() 
nll = torch.nn.BCEWithLogitsLoss()

initial_params = original.initial_params

predict = original.predict

negative_log_likelihood = original.negative_log_likelihood

def L2(X):
    return 1 / 100 * torch.sum(X ** 2)

def loss(X, y, weights):
    return negative_log_likelihood(X, y, weights) + L2(weights)

def gradient(X, y, weights):
    X, y, weights = to_torch(X, y, weights)
    weights.requires_grad_()
    neg = loss(X, y, weights)
    neg.backward()
    return weights.grad.numpy()

def forward(X, weights):
    return sigmoid(X @ weights)
