"""This file compares multinomial regression and 2-hidden-layers neural network"""
import numpy as np
from algorithms.accelerated_GD_adaptive import Nesterov_acceleration_adaptive
from algorithms.adam import Adam
from algorithms.gradient_descent_result import GradientDescentResult
from analysis.gradient_descent_result_plotting import GradientDescentResultPlotter
from datasets.mnist.files import mnist_test_X_y, mnist_train_X_y
from models import one_hidden_relu_softmax, one_hidden_relu_softmax_L2, softmax_regression
from test_runner.test_runner_file import Runner

np.random.seed(0)

X_train, y_train = mnist_train_X_y()

X_test, y_test = mnist_test_X_y()


iterations = 4000

startw = softmax_regression.initial_params(X_train, y_train)
test_set = {
    "w0": startw,
    # 'GD_params': {'step_size': 0.1},
    "GD_params": {"w0": startw, "start_alpha": [0.08]},
    "alg": [Nesterov_acceleration_adaptive],
    "model": softmax_regression,
    "max_iter": iterations,
    "data_set": (X_train, y_train),
    "test_set": (X_test, y_test),
    "epsilon": 0,
    "auto_stop": False,
    "batch": None,
}
runner = Runner(dic=test_set)
logistic_mnist_result: GradientDescentResult = runner.get_result()[0]


print("Logistic regression accuracy", logistic_mnist_result.get_best_accuracy())

K = 10
w0 = one_hidden_relu_softmax.initial_params(X_train.shape[1], 300, K)

test_set = {
    "w0": w0,
    # 'GD_params': {'L': [0.1], 'w0': w0},
    "GD_params": {"step_size": [0.08]},
    # 'GD_params': {'L': [0.01], 'w0': w0},
    "alg": [Adam],
    "model": one_hidden_relu_softmax,
    "max_iter": iterations,
    "data_set": (X_train, y_train),
    "test_set": (X_train, y_train),
    "epsilon": 0,
    "auto_stop": False,
    "batch": 64,
}
runner = Runner(dic=test_set)
one_hidden_result: GradientDescentResult = runner.get_result()[0]
print("One-hidden accuracy", one_hidden_result.get_best_accuracy())

#GradientDescentResultPlotter([one_hidden_result]).plot_accuracies_over_time().plot_function(
#    lambda x: logistic_mnist_result.get_accuracy_over_time()[x]
#).legend_placed("center right").with_result_labelled(
#    ["NN 1-hidden"]
#).with_functions_labelled(
#    ["Logistic regression - NAG"]
#).plot()




test_set = {
    "w0": w0,
    # 'GD_params': {'L': [0.1], 'w0': w0},
    "GD_params": {"step_size": [0.08]},
    # 'GD_params': {'L': [0.01], 'w0': w0},
    "alg": [Adam],
    "model": one_hidden_relu_softmax_L2,
    "max_iter": iterations,
    "data_set": (X_train, y_train),
    "test_set": (X_train, y_train),
    "epsilon": 0,
    "auto_stop": False,
    "batch": 64,
}
runner = Runner(dic=test_set)
one_hidden_L2_result: GradientDescentResult = runner.get_result()[0]
print("One-hidden accuracy, L2", one_hidden_L2_result.get_best_accuracy())

GradientDescentResultPlotter([one_hidden_result, one_hidden_L2_result]).plot_accuracies_over_time().plot_function(
    lambda x: logistic_mnist_result.get_accuracy_over_time()[x]
).legend_placed("center right").with_result_labelled(
    ["NN 1-hidden", "NN 1-hidden L2"]
).with_functions_labelled(
    ["Logistic regression - NAG"]
).plot()

print(one_hidden_L2_result.get_final_derivation())