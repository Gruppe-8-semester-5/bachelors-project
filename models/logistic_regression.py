import numpy as np
from models.utility import sigmoid

def predict(w:np.ndarray, features: np.ndarray):
    """The logistic regression prediction for a single point"""
    return sigmoid(w.transpose() * features)