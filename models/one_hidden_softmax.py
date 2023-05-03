import torch
import numpy as np

softmax = torch.nn.Softmax(dim = 1) 

# Torch:
    #    array([0.49014083, 0.10475402, 0.15254662, 0.58888827, 0.39610148,
    #           0.42369222, 0.50813386, 0.66072705, 0.49932855, 0.3417715 ])],
    # acc: 0.15015
# Own:
    #    array([0.49014492, 0.10474854, 0.15254679, 0.58888932, 0.3961015 ,
    #           0.42369211, 0.50813146, 0.6607279 , 0.49932947, 0.34177238])],
    # acc: 0.11608333333333332

# Torch on 100:
    #    array([0.49009353, 0.10480998, 0.15254704, 0.58888456, 0.39609421,
    #           0.42369438, 0.5081452 , 0.66072755, 0.49932237, 0.34176557])],

# Own on 100:
    #    array([0.49014543, 0.10474835, 0.15254655, 0.58888986, 0.39610125,
    #           0.42369189, 0.50813167, 0.66072775, 0.49932943, 0.3417722 ])],

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
    # # Torch works identically to this
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