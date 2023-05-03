import torch
import numpy as np

softmax = torch.nn.Softmax(dim = 1) 


def initial_params(input_dim, output_dim):
    W = np.random.rand(input_dim, output_dim)
    b = np.random.rand(output_dim)
    return np.array([W, b], dtype=object)

def forward(X, w, b):
    return softmax(X @ w + b)

def get_params(weights):
    w = weights[0]
    b = weights[1]
    return w, b

def to_torch(*vals):
    return map(torch.from_numpy, vals)

def predict(w:np.ndarray, features: np.ndarray):
    """The logistic regression prediction for a single point"""
    w, b = get_params(w)
    # print(softmax(forward(*to_torch(features, w, b))))
    return torch.argmax(softmax(forward(*to_torch(features, w, b))), dim=1).numpy()
    # exit()
    # return sigmoid(w.transpose() @ features)


def negative_log_likelihood(X, y, w, b):
    pred = forward(X, w, b)
    loss = torch.nn.NLLLoss()
    return loss(pred, y.long())
    # # Maybe fix this.
    # n = X.size(dim=0)
    # return 1 / n * torch.sum(torch.log(torch.nn.functional.one_hot(y.long()).transpose(0,1).double() @ pred))

def gradient(X, y, weights):
    X, y, w, b = to_torch(X, y, *get_params(weights))
    # X, y, w, b = to_torch(X, y.astype(float), *get_params(weights))
    # X = torch.from_numpy(X)
    # y = torch.from_numpy(y.astype(float))
    # w_, b_ = get_params(weights)
    # w = torch.from_numpy(w_)
    # b = torch.from_numpy(b_)
    w.requires_grad_()
    b.requires_grad_()
    nll = negative_log_likelihood(X, y, w, b)
    nll.backward()
    # exit()
    return np.array([w.grad.numpy(), b.grad.numpy()], dtype=object)
    # return weights.grad.numpy()