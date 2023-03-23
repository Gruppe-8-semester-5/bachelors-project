
import numpy as np


def lipschitz_neg_log_likelihood(feature_set: np.ndarray):
    """
    The Lipschitz constant, L, for maximum likelihood is
    L=(1/2n)\sum_{i=1}^n ||{x_i}||^2

    Use feature set X as input to compute it's lipschitz constant
    """
    feature_norm_sum = 0
    for features in feature_set:
        feature_norm_sum += np.linalg.norm(features)
    
    scale = 1/(2*len(feature_set))

    return scale * feature_norm_sum
