import numpy as np


def euclid_distance(a: np.ndarray, b: np.ndarray) -> float:
    """The euclidean distance. a and b should be vectors, or points in n-dim space"""
    sum_sq = 0.0
    for i in range(len(a)):
        diff = a[i] - b[i]
        sum_sq += diff**2
    return np.sqrt(sum_sq)