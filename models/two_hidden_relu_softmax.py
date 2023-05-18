from typing import TypedDict
import torch
import numpy as np

softmax = torch.nn.Softmax(dim = 1) 
relu = torch.nn.ReLU()

class WeightDict(TypedDict):
    W1: torch.Tensor
    b1: torch.Tensor
    W2: torch.Tensor
    b2: torch.Tensor


def initial_params(input_dim, hidden_layer, output_dim):
    W1 = np.random.normal(size=(input_dim, hidden_layer))
    b1 = np.random.normal(size=(hidden_layer))
    W2 = np.random.normal(size=(hidden_layer, output_dim))
    b2 = np.random.normal(size=(output_dim))
    return np.array([W1, b1, W2, b2], dtype=object)

def forward(X, w: WeightDict):
    return softmax(relu(X @ w['W1'] + w['b1']) @ w['W2'] + w['b2'])

def dict_to_torch(dic):
    return {k: torch.from_numpy(v) for k, v in dic.items()}

def get_params(weights) -> WeightDict:
    return dict_to_torch({'W1': weights[0], 'b1': weights[1], 'W2': weights[2], 'b2': weights[3]})
    # w1 = weights[0]
    # b1 = weights[1]
    # w2 = weights[2]
    # b2 = weights[3]
    # return w1, b1, w2, b2

def to_torch(*vals):
    return map(torch.from_numpy, vals)

def predict(w:np.ndarray, features: np.ndarray):
    """The logistic regression prediction for a single point"""
    w = get_params(w)
    # print(softmax(forward(*to_torch(features, w, b))))
    return torch.argmax(softmax(forward(*to_torch(features), w)), dim=1).numpy()
    # exit()
    # return sigmoid(w['transpose() @ features)


def negative_log_likelihood(X, y, w):
    pred = torch.log(forward(X, w))
    loss = torch.nn.NLLLoss()
    return loss(pred, y.long())

def gradient(X, y, weights):
    X, y = to_torch(X, y)
    w = get_params(weights)
    w['W1'].requires_grad_()
    w['b1'].requires_grad_()
    w['W2'].requires_grad_()
    w['b2'].requires_grad_()
    nll = negative_log_likelihood(X, y, w)
    nll.backward()
    # print(w['b2'].grad.numpy().any())
    # print(k)
    arr = [w['W1'], w['b1'], w['W2'], w['b2']]
    res_arr = [x.grad.numpy() for x in arr]
    return np.array(res_arr, dtype=object)
