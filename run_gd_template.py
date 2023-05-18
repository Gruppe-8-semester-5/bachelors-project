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
# import models.logistic_torch as cur_model
# import models.logistic_regression as normal_logistic
# import models.one_hidden_softmax as cur_model
import models.two_hidden_relu_softmax as cur_model
# Adam, two_hidden, 1000 iterations: 0.8255333333333333
# Adam, one_hidden, 1000 iterations: 0.7707833333333334
# import models.logistic_torch as cur_model
# from models.logistic_regression import logistic_regression
# from models.logistic_regression import gradient
# from models.logistic_regression import predict

epsilon=1.0e-10
iterations = 14

X, y = mnist_train_X_y()

# X, y = wine_X_y()
# import torch
# X = torch.nn.functional.normalize(torch.from_numpy(X), dim=1).numpy()
n = X.shape[0]

np.random.seed(0)
output_shape = 0
output_shape = np.amax(y) + 1

# w0 = cur_model.initial_params(X.shape[1], output_shape)
w0 = cur_model.initial_params(X.shape[1], 100, output_shape)
# w0 = cur_model.initial_params(X)
# grad = lambda w: gradient(X, y, w)
# List of things we want to test. Form (optimizer, params)
test_set = {
    'w0': w0,
    'GD_params': {'step_size': [0.01]},
    # 'GD_params': {'L': [0.01], 'w0': w0},
    'alg': [Adam],
    'model': cur_model,
    'max_iter': iterations,
    'data_set': (X, y),
    'epsilon':epsilon,
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
results = runner.get_result()
print(results[0].get_best_accuracy())
exit()
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

results = runner.get_result(alg=Standard_GD)