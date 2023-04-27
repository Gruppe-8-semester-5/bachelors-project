import numpy as np
from algorithms import GradientDescentResult, gradient_descent_template, standard_GD
from algorithms.standard_GD import Standard_GD
from algorithms.momentum_GD import Momentum
from models.logistic_regression import gradient
from datasets.winequality.files import wine_X_y
from analysis.lipschitz import lipschitz_binary_neg_log_likelihood
from models.logistic_regression import predict
from test_runner.test_runner_file import Runner

epsilon=1.0e-2
iterations = 100

def make_predictions(weights, wines):
    predictions = []
    for wine in wines:
        predictions.append(predict(weights, wine))
    return predictions

X, y = wine_X_y()
n = X.shape[0]

w0 = np.random.rand(X.shape[1])
# grad = lambda w: gradient(X, y, w)
# List of things we want to test. Form (optimizer, params)

test_set = {
    'w0': w0,
    'GD_params': {'step_size': [0.01, 0.05, 0.1, 0.5, 1]},
    'alg': [Standard_GD, Momentum],
    'derivation': gradient,
    'predictor': make_predictions,
    'max_iter': iterations,
    'data_set': (X, y),
    'epsilon':1.0e-2,
    'batch': None
}

# runner = Runner(dic = test_set)


runner = Runner(dic = test_set)

results = runner.get_res(alg=Standard_GD)
for r in results:
    print(r.to_serialized())
