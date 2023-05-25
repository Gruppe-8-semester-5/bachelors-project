"""This file compares multinomial regression and 2-hidden-layers neural network"""
import numpy as np
from algorithms.accelerated_GD_adaptive import Nesterov_acceleration_adaptive
from algorithms.adam import Adam
from algorithms.gradient_descent_result import GradientDescentResult
from analysis.gradient_descent_result_plotting import GradientDescentResultPlotter
from models import softmax_regression, twelve_hidden_relu_softmax
from models.utility import make_train_and_test_sets
from test_runner.test_runner_file import Runner
from torchvision import transforms
from datasets.fashion_mnist.files import fashion_mnist_X_y

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,),
            (0.5,),
        ),
    ]
)
np.random.seed(0)

X, y = fashion_mnist_X_y()
(X_train, y_train), (X_test, y_test) = make_train_and_test_sets(X, y, 0.8)


iterations = 200

startw = softmax_regression.initial_params(X_train, y_train)
test_set = {
    "w0": startw,
    # 'GD_params': {'step_size': 0.1},
    "GD_params": {"w0": startw, "start_alpha": [0.1]},
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
logistic_regression_mnist: GradientDescentResult = runner.get_result()[0]

K = 10
best_accuracy = 0
best_seed = 0
for rand_seed in range(10,40):
    np.random.seed(rand_seed)
    w0 = twelve_hidden_relu_softmax.initial_params(
        X_train.shape[1],
        [500, 500, 500, 200, 200, 200, 200, 200, 200, 200, 200, 200],
        K
    )
    test_set = {
        "w0": w0,
        # 'GD_params': {'L': [0.1], 'w0': w0},
        "GD_params": {"step_size": [0.1]},
        # 'GD_params': {'L': [0.01], 'w0': w0},
        "alg": [Adam],
        "model": twelve_hidden_relu_softmax,
        "max_iter": iterations,
        "data_set": (X_train, y_train),
        "test_set": (X_train, y_train),
        "epsilon": 0,
        "auto_stop": False,
        "batch": 64,
    }
    runner = Runner(dic=test_set)
    nn_result: GradientDescentResult = runner.get_result()[0]
    nn_result_accuracy = nn_result.get_best_accuracy()
    if best_accuracy < nn_result_accuracy:
        best_accuracy = nn_result_accuracy
        best_seed = rand_seed

print(best_seed, best_accuracy)