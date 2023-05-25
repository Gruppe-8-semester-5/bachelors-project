import numpy as np
from algorithms.accelerated_GD_adaptive import Nesterov_acceleration_adaptive
from algorithms.adam import Adam
from algorithms.gradient_descent_result import GradientDescentResult
from algorithms.gradient_descent_template_batch import find_minima
from analysis.gradient_descent_result_plotting import GradientDescentResultPlotter
from algorithms.standard_GD import Standard_GD
from datasets.mnist.files import mnist_X_y, mnist_X_y_simpel
from models.utility import make_train_and_test_sets
from test_runner.test_runner_file import Runner
from models import one_hidden_relu_softmax, convolution

np.random.seed(0)
X, y = mnist_X_y_simpel()
(X_train, y_train), (X_test, y_test) = make_train_and_test_sets(X, y, 0.8)

find_minima(Standard_GD(), X_train, y_train, None, 0, 4000, False, one_hidden_relu_softmax, 100, X_test, y_test)
# (X_train, y_train), (X_test, y_test) = make_train_and_test_sets(X, y, 0.8)

exit()
iterations = 100
np.random.seed(0)


startw = convolution.initial_params()
test_set = {
    "w0": startw,
    'GD_params': {'step_size': 0.1},
    # "GD_params": {"L": [0.1], "w0": startw},
    "alg": [Standard_GD],
    "model": convolution,
    "max_iter": iterations,
    "data_set": (X, y),
    # "data_set": (X_train, y_train),
    # "test_set": (X_test, y_test),
    "epsilon": 0,
    "auto_stop": False,
    "batch": None,
}
runner = Runner(dic=test_set)
conv_res: GradientDescentResult = runner.get_result()[0]

print("Convolution loss", conv_res.get_losses_over_time())


print("Convolution accuracy", conv_res.get_best_accuracy())
plotter = GradientDescentResultPlotter(
    [conv_res]
).plot_accuracies_over_time()
plotter.plot()

# predictions = convolution.predict(
#     conv_res.get_best_weights_over_time()[-1], X_train
# )
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
print("Logistic regression accuracy", conv_res.get_best_accuracy())
GradientDescentResultPlotter([nn_result]).plot_accuracies_over_time().plot_function(
    lambda x: conv_res.get_accuracy_over_time()[x]
).legend_placed("center right").with_result_labelled(
    ["Neural network"]
).with_functions_labelled(
    ["Logistic regression - NAG"]
).plot()
