"""This file compares multinomial regression and 2-hidden-layers neural network"""
import numpy as np
from algorithms.accelerated_GD_adaptive import Nesterov_acceleration_adaptive
from algorithms.adam import Adam
from algorithms.gradient_descent_result import GradientDescentResult
from algorithms.heavy_ball import Heavy_ball
from algorithms.standard_GD import Standard_GD
from analysis.gradient_descent_result_plotting import GradientDescentResultPlotter
from datasets.mnist.files import mnist_test_X_y, mnist_train_X_y
from models import one_hidden_relu_softmax, one_hidden_relu_softmax_L2, softmax_regression
from test_runner.test_runner_file import Runner

np.random.seed(0)

X_train, y_train = mnist_train_X_y()

X_test, y_test = mnist_test_X_y()


iterations = 500

startw = softmax_regression.initial_params(X_train, y_train)
test_set = {
    "w0": startw,
    # 'GD_params': {'step_size': 0.1},
    "GD_params": {"w0": startw, "start_alpha": [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 0.8]},
    "alg": [Nesterov_acceleration_adaptive],
    "model": softmax_regression,
    "max_iter": iterations,
    "data_set": (X_train, y_train),
    "test_set": (X_test, y_test),
    "epsilon": 0,
    "auto_stop": False,
    "batch": None,   
    "accuracy_compute_interval": 10
}
runner = Runner(dic=test_set)
nesterov_test_results = runner.get_result()

GradientDescentResultPlotter(nesterov_test_results).legend_placed("upper right").with_result_labelled(["alpha=0.01", "alpha=0.05", "alpha=0.1", "alpha=0.15", "alpha=0.2", "alpha=0.5", "alpha=0.8"]).plot_with_y_axis_logarithmic().plot_loss().plot()

test_set = {
    "w0": startw,
    # 'GD_params': {'step_size': 0.1},
    "GD_params": {"w0": startw, "step_size": [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 0.8]},
    "alg": [Heavy_ball],
    "model": softmax_regression,
    "max_iter": iterations,
    "data_set": (X_train, y_train),
    "test_set": (X_test, y_test),
    "epsilon": 0,
    "auto_stop": False,
    "batch": None,   
    "accuracy_compute_interval": 10
}
runner = Runner(dic=test_set)
heavy_ball_test_results = runner.get_result()

GradientDescentResultPlotter(heavy_ball_test_results).legend_placed("upper right").with_result_labelled(["alpha=0.01", "alpha=0.05", "alpha=0.1", "alpha=0.15", "alpha=0.2", "alpha=0.5", "alpha=0.8"]).plot_with_y_axis_logarithmic().plot_loss().plot()


test_set = {
    "w0": startw,
    # 'GD_params': {'step_size': 0.1},
    "GD_params": {"step_size": [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 0.8]},
    "alg": [Standard_GD],
    "model": softmax_regression,
    "max_iter": iterations,
    "data_set": (X_train, y_train),
    "test_set": (X_test, y_test),
    "epsilon": 0,
    "auto_stop": False,
    "batch": None,   
    "accuracy_compute_interval": 10
}
runner = Runner(dic=test_set)
standard_gd_test_results = runner.get_result()

GradientDescentResultPlotter(standard_gd_test_results).legend_placed("upper right").with_result_labelled(["alpha=0.01", "alpha=0.05", "alpha=0.1", "alpha=0.15", "alpha=0.2", "alpha=0.5", "alpha=0.8"]).plot_with_y_axis_logarithmic().plot_loss().plot()




GradientDescentResultPlotter(nesterov_test_results).legend_placed("upper right").with_result_labelled(["alpha=0.01", "alpha=0.05", "alpha=0.1", "alpha=0.15", "alpha=0.2", "alpha=0.5", "alpha=0.8"]).plot_distance_to_absolute_most_accurate_weight().plot()
GradientDescentResultPlotter(heavy_ball_test_results).legend_placed("upper right").with_result_labelled(["alpha=0.01", "alpha=0.05", "alpha=0.1", "alpha=0.15", "alpha=0.2", "alpha=0.5", "alpha=0.8"]).plot_distance_to_absolute_most_accurate_weight().plot()
GradientDescentResultPlotter(standard_gd_test_results).legend_placed("upper right").with_result_labelled(["alpha=0.01", "alpha=0.05", "alpha=0.1", "alpha=0.15", "alpha=0.2", "alpha=0.5", "alpha=0.8"]).plot_distance_to_absolute_most_accurate_weight().plot()

GradientDescentResultPlotter([nesterov_test_results[5], heavy_ball_test_results[5], standard_gd_test_results[6]]).legend_placed("lower right").with_result_labelled(["Nesterov", "Momentum/Heavy ball", "No momentum"]).plot_accuracies_over_time().plot()
