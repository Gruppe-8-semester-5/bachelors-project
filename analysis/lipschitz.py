import numpy as np


def lipschitz_binary_neg_log_likelihood(feature_set: np.ndarray, labels: np.ndarray):
    """
    The Lipschitz constant, L, for maximum likelihood is
    L=(1/4n)\sum_{i=1}^n ||{x_i}||^2

    Use feature set X as input to compute it's lipschitz constant. 
    Can only be used for binary classification models! todo: maybe change to allow for intervals
    """
    labels = [0. if l <= 0 else 1 for l in labels]

    feature_norm_sum = 0
    n = len(feature_set)
    for i in range(0, n):
        features = feature_set[i]
        feature_norm_sum += np.linalg.norm(features)**2

    scale = 1/(4*n)

    return scale * feature_norm_sum

def lipschitz_binary_neg_log_likelihood_regularized(feature_set: np.ndarray, labels: np.ndarray, weight_factor: float = 0.01):
    """
    2 + binary neg log.
    """
    return weight_factor * 2 + lipschitz_binary_neg_log_likelihood(feature_set, labels)
