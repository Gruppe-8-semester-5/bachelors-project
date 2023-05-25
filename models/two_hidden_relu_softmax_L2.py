from typing import TypedDict
import torch
import numpy as np

softmax = torch.nn.Softmax(dim = 1) 
log_softmax = torch.nn.LogSoftmax(dim = 1)
relu = torch.nn.ReLU()

class WeightDict(TypedDict):
    W1: torch.Tensor
    b1: torch.Tensor
    W2: torch.Tensor
    b2: torch.Tensor
    W3: torch.Tensor
    b3: torch.Tensor


def initial_params(input_dim, hidden_layer1, hidden_layer2, output_dim):
    W1 = np.random.normal(size=(input_dim, hidden_layer1))
    b1 = np.random.normal(size=(hidden_layer1))
    W2 = np.random.normal(size=(hidden_layer1, hidden_layer2))
    b2 = np.random.normal(size=(hidden_layer2))
    W3 = np.random.normal(size=(hidden_layer2, output_dim))
    b3 = np.random.normal(size=(output_dim))
    return np.array([W1, b1, W2, b2, W3, b3], dtype=object)

def forward(X, w: WeightDict):
    return softmax(fw(X, w))

def fw(X, w: WeightDict):
    L2_lda = 0.0001
    return relu(relu(X @ (w['W1'] + (L2_lda * w['W1']**2)) + w['b1']) @ (w['W2'] + (L2_lda * w['W2']**2)) + w['b2']) @ (w['W3'] + (L2_lda * w['W3']**2)) + w['b3']

def dict_to_torch(dic):
    return {k: torch.from_numpy(v) for k, v in dic.items()}

def get_params(weights) -> WeightDict:
    return dict_to_torch({'W1': weights[0], 'b1': weights[1], 'W2': weights[2], 'b2': weights[3], 'W3': weights[4], 'b3': weights[5]})

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
    pred = log_softmax(fw(X, w))
    loss = torch.nn.NLLLoss()
    return loss(pred, y.long())

def gradient(X, y, weights):
    return gradient_and_loss(X, y, weights)[0]

def gradient_and_loss(X, y, weights):
    X, y = to_torch(X, y)
    w = get_params(weights)
    w['W1'].requires_grad_()
    w['b1'].requires_grad_()
    w['W2'].requires_grad_()
    w['b2'].requires_grad_()
    w['W3'].requires_grad_()
    w['b3'].requires_grad_()
    nll = negative_log_likelihood(X, y, w)
    nll.backward()
    arr = [w['W1'], w['b1'], w['W2'], w['b2'], w['W3'], w['b3']]
    res_arr = [x.grad.numpy() for x in arr]
    return np.array(res_arr, dtype=object), nll.item()
