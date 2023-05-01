import numpy as np
from algorithms import GradientDescentResult, gradient_descent_template, standard_GD
from algorithms.standard_GD import Standard_GD
from algorithms.momentum_GD import Momentum
from datasets.winequality.files import wine_X_y
from analysis.lipschitz import lipschitz_binary_neg_log_likelihood
from test_runner.test_runner_file import Runner
from models.logistic_regression import gradient as g
# from models.logistic_regression import gradient
# from models.logistic_regression import predict

from models.simple_nn import gradient
from models.simple_nn import predict

epsilon=1.0e-2
iterations = 100

def make_predictions(weights, wines):
    predictions = []
    for wine in wines:
        predictions.append(predict(weights, wine))
    return predictions

X, y = wine_X_y()
n = X.shape[0]

np.random.seed(0)
w0 = np.random.rand(X.shape[1])
# grad = lambda w: gradient(X, y, w)
# List of things we want to test. Form (optimizer, params)

test_set = {
    'w0': w0,
    'GD_params': {'step_size': [0.01]},
    'alg': [Standard_GD],
    'derivation': gradient,
    'predictor': make_predictions,
    'max_iter': iterations,
    'data_set': (X, y),
    'epsilon':1.0e-10,
    'batch': None
}
# test_set = {
#     'w0': w0,
#     'GD_params': {'step_size': [0.01, 0.05, 0.1, 0.5, 1]},
#     'alg': [Standard_GD, Momentum],
#     'derivation': gradient,
#     'predictor': make_predictions,
#     'max_iter': iterations,
#     'data_set': (X, y),
#     'epsilon':1.0e-2,
#     'batch': None
# }

# print(gradient(np.array([2.]), np.array([0]), np.array([1.])))

# runner = Runner(dic = test_set)


runner = Runner(dic = test_set)

results = runner.get_res(alg=Standard_GD)
results = runner.get_res()
# for r in results:
#     print(r.to_serialized())

test_set = {
    'w0': w0,
    'GD_params': {'step_size': [0.01, 0.05, 0.1, 0.5, 1]},
    'alg': [Standard_GD, Momentum],
    'derivation': g,
    'predictor': make_predictions,
    'max_iter': iterations,
    'data_set': (X, y),
    'epsilon':1.0e-2,
    'batch': None
}

runner = Runner(dic = test_set)

results = runner.get_res(alg=Standard_GD)