from typing import TypedDict
import torch
import numpy as np

from models.utility import to_torch

softmax = torch.nn.Softmax(dim = 1) 
log_softmax = torch.nn.LogSoftmax(dim = 1)
relu = torch.nn.ReLU()
channels1 = 16
channels2 = 32
final = channels2 * 9
out = 10
# conv1 = torch.nn.Conv2d(in_channels=1, out_channels=channels1, kernel_size=5, stride=4, padding=2, dtype=torch.double)
# conv2 = torch.nn.Conv2d(in_channels=channels1, out_channels=channels2, kernel_size=5, stride=2, padding=2, dtype=torch.double)
pool1 = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=1)
pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2,padding=1)


# print(conv1.weight.shape)
# print(conv2.weight.shape)
# RuntimeError: mat1 and mat2 shapes cannot be multiplied (70000x128 and 4x10)

class WeightDict(TypedDict):
    C1: torch.Tensor
    C2: torch.Tensor
    F: torch.Tensor


def initial_params():
    C1 = np.random.normal(size=(16, 1, 5, 5))
    C2 = np.random.normal(size=(32, 16, 5, 5))
    F = np.random.normal(size=(out, final))
    return np.array([C1, C2, F], dtype=object)

def forward(X, w: WeightDict):

    return softmax(fw(X, w))

def fw(X, w: WeightDict):
    conv1 = w['C1']
    conv2 = w['C2']
    full = w['F']
    res = pool2(relu(conv2(pool1(relu(conv1(X))))))
    return full(res.view(res.size(0), -1) )

def dict_to_torch(dic):
    return {k: torch.from_numpy(v) for k, v in dic.items()}

def get_params(weights) -> WeightDict:
    C1, C2, F = to_torch(weights[0], weights[1], weights[2])
    conv1 = torch.nn.Conv2d(in_channels=1, out_channels=channels1, kernel_size=5, stride=2, padding=2, dtype=torch.double)
    conv2 = torch.nn.Conv2d(in_channels=channels1, out_channels=channels2, kernel_size=5, stride=2, padding=2, dtype=torch.double)
    full = torch.nn.Linear(in_features=channels2 * final, out_features=out, dtype=torch.double)
    conv1.weight = torch.nn.Parameter(C1)
    conv2.weight = torch.nn.Parameter(C2)
    full.weight = torch.nn.Parameter(F)
    return {'C1': conv1, 'C2': conv2, 'F': full}

def predict(w:np.ndarray, features: np.ndarray):
    """The logistic regression prediction for a single point"""
    w = get_params(w)
    return torch.argmax(softmax(fw(*to_torch(features), w)), dim=1).numpy()

def negative_log_likelihood(X, y, w):
    pred = log_softmax(fw(X, w))
    loss = torch.nn.NLLLoss()
    return loss(pred, y.long())

def gradient(X, y, weights):
    return gradient_and_loss(X, y, weights)[0]

def gradient_and_loss(X, y, weights):
    X, y = to_torch(X, y)
    w = get_params(weights)
    # w['W1'].requires_grad_()
    # w['b1'].requires_grad_()
    # w['W2'].requires_grad_()
    # w['b2'].requires_grad_()
    # w['W3'].requires_grad_()
    # w['b3'].requires_grad_()
    nll = negative_log_likelihood(X, y, w)
    nll.backward()
    arr = [w['C1'], w['C2'], w['F']]
    res_arr = [x.weight.grad.numpy() for x in arr]
    return np.array(res_arr, dtype=object), nll.item()
