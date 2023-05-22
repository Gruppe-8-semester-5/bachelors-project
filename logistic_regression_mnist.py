import math
import random
import matplotlib.pyplot as plt
import numpy as np
from algorithms.accelerated_GD_adaptive import Nesterov_acceleration_adaptive
from algorithms.adam import Adam
from analysis.gradient_descent_result_plotting import GradientDescentResultPlotter
from datasets.mnist.files import mnist_test_X_y, read_train_data, read_test_data, mnist_train_X_y
from datasets.winequality.wine import Wine
from models.logistic_regression import gradient_k, predict_with_softmax
from algorithms import GradientDescentResult, gradient_descent_template
from algorithms.standard_GD import Standard_GD
from algorithms.momentum_GD import Momentum
from algorithms.accelerated_GD import Nesterov_acceleration
from models.utility import accuracy_k_encoded
from models import logistic_torch, multinomial_logistic_regression, two_hidden_relu_softmax
from test_runner.test_runner_file import Runner
from sklearn.preprocessing import PolynomialFeatures
np.random.seed(2)

X_train, y_train = mnist_train_X_y()
X_test, y_test = mnist_test_X_y()


K = 10

iterations = 2000
np.random.seed(123)
w0 = two_hidden_relu_softmax.initial_params(X_train.shape[1], 50, K)

startw = logistic_torch.initial_params(X_train)
test_set = {
    'w0': startw,
    'GD_params': {'w0': startw, 'start_alpha':0.01},
    # 'GD_params': {'L': [0.01], 'w0': w0},
    'alg': [Standard_GD],
    'model': logistic_torch,
    'max_iter': iterations,
    'data_set': (X_train, y_train),
    'test_set': (X_test, y_test),
    'epsilon':0,
    'auto_stop': False,
    'batch': None
}
runner = Runner(dic = test_set)
logistic_regression_mnist: GradientDescentResult = runner.get_result()[0]


print("Logistic regression accuracy", logistic_regression_mnist.get_best_accuracy())
GradientDescentResultPlotter([logistic_regression_mnist]).plot_accuracies_over_time().plot()

test_set = {
    'w0': w0,
    'GD_params': {'step_size': [0.01]},
    # 'GD_params': {'L': [0.01], 'w0': w0},
    'alg': [Standard_GD],
    'model': two_hidden_relu_softmax,
    'max_iter': iterations,
    'data_set': (X_train, y_train),
    'test_set': (X_test, y_test),
    'epsilon':0,
    'auto_stop': False,
    'batch': None
}
runner = Runner(dic = test_set)
nn_result: GradientDescentResult = runner.get_result()[0]
print("Logistic regression accuracy", logistic_regression_mnist.get_best_accuracy())
GradientDescentResultPlotter([nn_result]).plot_accuracies_over_time().plot_function(lambda x: logistic_regression_mnist.get_accuracy_over_time()[x]).legend_placed("center right").with_result_labelled(["Neural network"]).with_functions_labelled(["Logistic regression - NAG"]).plot()