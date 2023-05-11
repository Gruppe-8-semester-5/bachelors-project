from models.logistic_regression import gradient_k, predict_with_softmax


def gradient(X, y, weights):
    return gradient_k(X, y, weights) + 2*weights

def predict(weights, X):
    return predict_with_softmax(X, weights)