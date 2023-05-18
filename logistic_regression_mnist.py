import math
import random
import matplotlib.pyplot as plt
import numpy as np
from algorithms.adam import Adam
from analysis.gradient_descent_result_plotting import GradientDescentResultPlotter
from datasets.mnist.files import read_train_data, read_test_data, mnist_train_X_y
from datasets.winequality.wine import Wine
from models.logistic_regression import gradient_k, predict_with_softmax
from algorithms import GradientDescentResult, gradient_descent_template
from algorithms.standard_GD import Standard_GD
from algorithms.momentum_GD import Momentum
from algorithms.accelerated_GD import Nesterov_acceleration
from models.utility import accuracy_k_encoded
from models import multinomial_logistic_regression, two_hidden_relu_softmax
from test_runner.test_runner_file import Runner
from sklearn.preprocessing import PolynomialFeatures
np.random.seed(0)

X_train, y_train = mnist_train_X_y(1000)

K = 10

iterations = 1000
runner_results = [] # Our results
input_dim = X_train.shape[1]
print(input_dim)
w0 = two_hidden_relu_softmax.initial_params(X_train.shape[1], 200, K)
print("HELLO W0")
test_set = {
    'w0': w0,
    'GD_params': {'step_size': [0.01]},
    # 'GD_params': {'L': [0.01], 'w0': w0},
    'alg': [Standard_GD],
    'model': two_hidden_relu_softmax,
    'max_iter': iterations,
    'data_set': (X_train, y_train),
    #'test_set': (feature_array_test, label_array_test),
    'epsilon':0,
    'auto_stop': False,
    'batch': None
}
print("HELLO test_set")
runner = Runner(dic = test_set)
print("HELLO runner")
std_gd_result: GradientDescentResult = runner.get_result()[0]
runner_results.append(std_gd_result)
print(std_gd_result.get_best_accuracy())
print("HELLO result")

exit()
