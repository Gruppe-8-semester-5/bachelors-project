import numpy as np
from algorithms.accelerated_GD_adaptive import Nesterov_acceleration_adaptive
from algorithms.adam import Adam
from algorithms.gradient_descent_result import GradientDescentResult
from analysis.gradient_descent_result_plotting import GradientDescentResultPlotter
from datasets.mnist.files import mnist_test_X_y, mnist_train_X_y
from algorithms.standard_GD import Standard_GD
from datasets.winequality.files import wine_X_y_quality
from models import one_hidden_relu_softmax, softmax_regression, two_hidden_relu_softmax
from models.utility import make_train_and_test_sets
from test_runner.test_runner_file import Runner
from torchvision import datasets, transforms
from datasets.fashion_mnist.files import fashion_mnist_X_y
import torch

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,),
            (0.5,),
        ),
    ]
)
np.random.seed(2)

X, y = fashion_mnist_X_y()
(X_train, y_train), (X_test, y_test) = make_train_and_test_sets(X, y, 0.8)


iterations = 250
np.random.seed(0)
startw = softmax_regression.initial_params(X_train, y_train)

test_set = {
    "w0": startw,
    # 'GD_params': {'step_size': 0.1},
    "GD_params": {"L": [0.1], "w0": startw},
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


print("Logistic regression accuracy", logistic_regression_mnist.get_best_accuracy())
# GradientDescentResultPlotter(
#     [logistic_regression_mnist]).plot_accuracies_over_time().plot()
print(
    softmax_regression.predict(
        logistic_regression_mnist.get_best_weights_over_time()[-1], X_train
    )
    == y_train
)


K = 10
comparison_results = []
for layer1_size in [10, 20, 30, 40, 50, 70, 90, 120, 160, 200, 300]:
    for layer2_size in [10, 20, 30, 40, 50, 70, 90, 120]:
        w0 = two_hidden_relu_softmax.initial_params(
            X_train.shape[1], layer1_size, layer2_size, K
        )
        test_set = {
            "w0": w0,
            # 'GD_params': {'L': [0.1], 'w0': w0},
            "GD_params": {"step_size": [0.1]},
            # 'GD_params': {'L': [0.01], 'w0': w0},
            "alg": [Adam],
            "model": two_hidden_relu_softmax,
            "max_iter": iterations,
            "data_set": (X_train, y_train),
            "test_set": (X_test, y_test),
            "epsilon": 0,
            "auto_stop": False,
            "batch": 64,
        }
        runner = Runner(dic=test_set)
        nn_result: GradientDescentResult = runner.get_result()[0]
        comparison_results.append(
            f"NN accuracy ({layer1_size},{layer2_size}) - {nn_result.get_best_accuracy()}"
        )
for res in comparison_results:
    print(res)
