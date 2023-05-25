import numpy as np
from algorithms.accelerated_GD_adaptive import Nesterov_acceleration_adaptive
from algorithms.adam import Adam
from algorithms.gradient_descent_result import GradientDescentResult
from analysis.gradient_descent_result_plotting import GradientDescentResultPlotter
from datasets.fashion_mnist.files import fashion_mnist_X_y_original
from algorithms.standard_GD import Standard_GD
from datasets.mnist.files import mnist_test_X_y, mnist_train_X_y
from models.utility import make_train_and_test_sets
# from models import one_hidden_relu_softmax, softmax_regression
from test_runner.test_runner_file import Runner
from models import convolution

np.random.seed(1)

(X_train, y_train) = mnist_train_X_y()

(X_test, y_test) = mnist_test_X_y()

iterations = 200
np.random.seed(3)

startw = convolution.initial_params()
test_set = {
    "w0": startw,
    'GD_params': {'step_size': 0.05},
    # "GD_params": {"L": [0.1], "w0": startw},
    "alg": [Adam],
    "model": convolution,
    "max_iter": iterations,
    "data_set": (X_train, y_train),
    "test_set": (X_test, y_test),
    "epsilon": 0,
    "auto_stop": False,
    "batch": None,
}
runner = Runner(dic=test_set)
logistic_regression_mnist: GradientDescentResult = runner.get_result()[0]


print("Logistic regression accuracy", logistic_regression_mnist.get_best_accuracy())
plotter = GradientDescentResultPlotter(
    [logistic_regression_mnist]
).plot_accuracies_over_time()
plotter.plot()

predictions = convolution.predict(
    logistic_regression_mnist.get_best_weights_over_time()[-1], X_train
)
print(predictions == y_train)
print(
    "Logistic regression accuracy",
    logistic_regression_mnist.get_accuracy_over_time()[-1],
)
exit()
K = 10
w0 = one_hidden_relu_softmax.initial_params(X_train.shape[1], 50, K)

test_set = {
    "w0": w0,
    "GD_params": {"step_size": [0.01]},
    # 'GD_params': {'L': [0.01], 'w0': w0},
    "alg": [Standard_GD],
    "model": one_hidden_relu_softmax,
    "max_iter": iterations,
    "data_set": (X_train, y_train),
    "test_set": (X_test, y_test),
    "epsilon": 0,
    "auto_stop": False,
    "batch": None,
}
runner = Runner(dic=test_set)
nn_result: GradientDescentResult = runner.get_result()[0]
print("Logistic regression accuracy", logistic_regression_mnist.get_best_accuracy())
GradientDescentResultPlotter([nn_result]).plot_accuracies_over_time().plot_function(
    lambda x: logistic_regression_mnist.get_accuracy_over_time()[x]
).legend_placed("center right").with_result_labelled(
    ["Neural network"]
).with_functions_labelled(
    ["Logistic regression - NAG"]
).plot()
