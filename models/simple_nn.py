import torch
import numpy as np

sigmoid = torch.nn.Sigmoid() 


softmax = torch.nn.Softmax()
cross = torch.nn.CrossEntropyLoss()



# https://stackoverflow.com/questions/60903821/how-to-prevent-inf-while-working-with-exponential
def log1pexp(x):
    # more stable version of log(1 + exp(x))
    return torch.where(x < 50, torch.log1p(torch.exp(x)), x)

def forward(X, y, weights):
    
    # z = X @ weights
    # p = torch.exp(z) / (1 + torch.exp(z))
    # p = 1 / (1 + torch.exp(-z))
    # y_ = torch.reshape(y, (1,n))
    # weights_ = torch.reshape(weights, (l,1))
    # nll = torch.sum(torch.log(1+torch.exp(-(2 * y.double() - 1) * (X @ weights))))
    # Seems to workk?
    zeros = torch.zeros_like(y)
    nll = torch.sum(torch.logaddexp(-(2 * y.double() - 1) * (X @ weights), zeros))
    # nll = torch.sum(log1pexp(-(2 * y.double() - 1) * (X @ weights)))
    # nll = 1 / weights.size()[0] *  torch.sum(torch.log(1+torch.exp(-(2 * y.double() - 1) * (X @ weights))))
    # nll = -torch.sum(y * torch.log(p) + (1 - y) * torch.log(1 - p))
    # nll = 1 / weights.size()[0] *  torch.sum(torch.log(1+torch.exp(-(2 * y.double() - 1) * (X @ weights))))
    if torch.isinf(nll).any():
        print(nll)
        print(-(2 * y.double() - 1) * (X @ weights))
        print('ERROR')
        exit()
        # import pdb; pdb.set_trace()
    return nll.nan_to_num(nan=0.0)


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
    # print(nll)
    # print(weights.grad)
    # print('Weights: ', weights)
    # print('Forward:', nll)
    # print('Gradient', weights.grad)
    # print('grad: ', weights.grad)
    # print(weights)
    return weights.grad.numpy()
    # X = torch.from_numpy(X)
    # y = torch.from_numpy(y)
    # z = X @ weights
    # p = 1 / (1 + torch.exp(-z))
    # grad = -X.T @ (y - p)
    # return grad.numpy()