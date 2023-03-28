import numpy as np


def sigmoid(z):
    return 1/(1+(np.exp(-z)))

def neg_log_likelihood_gradient(w: np.ndarray, features: np.ndarray, labels: np.ndarray):
    """The gradient for negative log-likelihood. Labels must be in {0, 1}"""
    assert len(features) == len(labels)
    n = len(features)


    sum = 0
    for i in range(0,n):
        x = features[i]
        y = labels[i]
        sum += -y*x*sigmoid(-y * w.transpose() * x)
    return (1/n) * sum