from typing import TypedDict
import torch
import numpy as np

softmax = torch.nn.Softmax(dim=1) 
log_softmax = torch.nn.LogSoftmax(dim=1)
relu = torch.nn.ReLU()


class WeightDict(TypedDict):
    W1: torch.Tensor
    b1: torch.Tensor
    W2: torch.Tensor
    b2: torch.Tensor
    W3: torch.Tensor
    b3: torch.Tensor
    W4: torch.Tensor
    b4: torch.Tensor
    W5: torch.Tensor
    b5: torch.Tensor
    W6: torch.Tensor
    b6: torch.Tensor
    W7: torch.Tensor
    b7: torch.Tensor
    W8: torch.Tensor
    b8: torch.Tensor
    W9: torch.Tensor
    b9: torch.Tensor
    W10: torch.Tensor
    b10: torch.Tensor
    W11: torch.Tensor
    b11: torch.Tensor
    W12: torch.Tensor
    b12: torch.Tensor
    W13: torch.Tensor
    b13: torch.Tensor


def initial_params(input_dim, layers: np.ndarray, output_dim):
    if len(layers) != 12:
        raise Exception("Cannot create neural network, needs 12 layer sizes specified")
    
    hidden_layer1 = layers[0]
    hidden_layer2 = layers[1]
    hidden_layer3 = layers[2]
    hidden_layer4 = layers[3]
    hidden_layer5 = layers[4]
    hidden_layer6 = layers[5]
    hidden_layer7 = layers[6]
    hidden_layer8 = layers[7]
    hidden_layer9 = layers[8]
    hidden_layer10 = layers[9]
    hidden_layer11 = layers[10]
    hidden_layer12 = layers[11]
    W1 = np.random.normal(size=(input_dim, hidden_layer1))
    b1 = np.random.normal(size=(hidden_layer1))
    W2 = np.random.normal(size=(hidden_layer1, hidden_layer2))
    b2 = np.random.normal(size=(hidden_layer2))
    W3 = np.random.normal(size=(hidden_layer2, hidden_layer3))
    b3 = np.random.normal(size=(hidden_layer3))
    W4 = np.random.normal(size=(hidden_layer3, hidden_layer4))
    b4 = np.random.normal(size=(hidden_layer4))
    W5 = np.random.normal(size=(hidden_layer4, hidden_layer5))
    b5 = np.random.normal(size=(hidden_layer5))
    W6 = np.random.normal(size=(hidden_layer5, hidden_layer6))
    b6 = np.random.normal(size=(hidden_layer6))
    W7 = np.random.normal(size=(hidden_layer6, hidden_layer7))
    b7 = np.random.normal(size=(hidden_layer7))
    W8 = np.random.normal(size=(hidden_layer7, hidden_layer8))
    b8 = np.random.normal(size=(hidden_layer8))
    W9 = np.random.normal(size=(hidden_layer8, hidden_layer9))
    b9 = np.random.normal(size=(hidden_layer9))
    W10 = np.random.normal(size=(hidden_layer9, hidden_layer10))
    b10 = np.random.normal(size=(hidden_layer10))
    W11 = np.random.normal(size=(hidden_layer10, hidden_layer11))
    b11 = np.random.normal(size=(hidden_layer11))
    W12 = np.random.normal(size=(hidden_layer11, hidden_layer12))
    b12 = np.random.normal(size=(hidden_layer12))
    W13 = np.random.normal(size=(hidden_layer12, output_dim))
    b13 = np.random.normal(size=(output_dim))

    return np.array([W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7, W8, b8, W9, b9, W10, b10, W11, b11, W12, b12, W13, b13], dtype=object)

def forward(X, w: WeightDict):
    return softmax(fw(X, w))

def fw(X, w: WeightDict):
    forward_computation = relu(X @ w['W1'] + w['b1'])

    for i in range(2, 13):
        forward_computation = relu(forward_computation @ w[f'W{i}'] + w[f'b{i}'])

    return forward_computation @ w['W13'] + w['b13']

def dict_to_torch(dic):
    return {k: torch.from_numpy(v) for k, v in dic.items()}

def get_params(weights) -> WeightDict:
    named_dict = {}
    for i in range(0, len(weights), 2):
        named_dict[f'W{round(i/2)+1}'] = weights[i]
        named_dict[f'b{round(i/2)+1}'] = weights[i + 1]
    return dict_to_torch(named_dict)


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
    for i in range(0, len(weights), 2):
        w[f'W{round(i/2)+1}'].requires_grad_()
        w[f'b{round(i/2)+1}'].requires_grad_()
    nll = negative_log_likelihood(X, y, w)
    nll.backward()
    arr = []

    for i in range(0, len(weights), 2):
        arr.append(w[f'W{round(i/2)+1}'])
        arr.append(w[f'b{round(i/2)+1}'])

    res_arr = [x.grad.numpy() for x in arr]
    return np.array(res_arr, dtype=object), nll.item()
