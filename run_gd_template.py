import numpy as np
from algorithms import GradientDescentResult, gradient_descent_template, standard_GD
from algorithms.standard_GD import Standard_GD
from algorithms.momentum_GD import Momentum
from algorithms.accelerated_GD import Nesterov_acceleration
from algorithms.adam import Adam
from datasets.mnist.files import mnist_train_X_y
from datasets.winequality.files import wine_X_y
from analysis.lipschitz import lipschitz_binary_neg_log_likelihood
from test_runner.test_runner_file import Runner
# from models.logistic_torch import logistic_regression_torch
import models.logistic_torch as torch_logistic
import models.logistic_regression as normal_logistic
import models.one_hidden_softmax as cur_model
# import models.logistic_torch as cur_model
# from models.logistic_regression import logistic_regression
# from models.logistic_regression import gradient
# from models.logistic_regression import predict

epsilon=1.0e-2
iterations = 100
one_in_k = False

X, y = mnist_train_X_y(one_in_k)

# X, y = wine_X_y()
n = X.shape[0]

np.random.seed(0)
output_shape = 0
if one_in_k:
    output_shape = y.shape[1]
else:
    output_shape = np.amax(y) + 1

# w0 = cur_model.initial_params(X.shape[1], y.shape[1])
w0 = cur_model.initial_params(X.shape[1], output_shape)
# w0 = cur_model.initial_params(X)
# grad = lambda w: gradient(X, y, w)
# List of things we want to test. Form (optimizer, params)

test_set = {
    'w0': w0,
    'GD_params': {'step_size': [0.01]},
    # 'GD_params': {'L': [0.01], 'w0': w0},
    'alg': [Nesterov_acceleration],
    'model': cur_model,
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
# print(gradient(X, y, w0))
# print(g(X, y, w0))
# import torch
# print(torch.sum(torch.logaddexp(torch.tensor([-1.0]), torch.tensor([-1.0, -2, -3]))))
# a = torch.randn(3, 3)
# print(torch.logsumexp(a, 0))
# exit()

runner = Runner(dic = test_set)

# results = runner.get_res(alg=Standard_GD)
results = runner.get_res()
print(results[0].get_best_accuracy())
# for r in results:
#     print(r.to_serialized())
exit()

test_set = {
    'w0': w0,
    'GD_params': {'step_size': [0.01, 0.05, 0.1, 0.5, 1]},
    'alg': [Standard_GD, Momentum],
    'model': normal_logistic,
    'max_iter': iterations,
    'data_set': (X, y),
    'epsilon':1.0e-2,
    'batch': None
}

runner = Runner(dic = test_set)

results = runner.get_res(alg=Standard_GD)