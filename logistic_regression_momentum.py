import numpy as np
from analysis.gradient_descent_result_plotting import GradientDescentResultPlotter
from datasets.winequality.files import wine_X_y_quality
from algorithms import GradientDescentResult
from algorithms.momentum_GD import Momentum
from models import softmax_regression
from models.utility import make_train_and_test_sets
from test_runner.test_runner_file import Runner

#for ra in range(0, 100):

X, y = wine_X_y_quality()

np.random.seed(1)
start_weight = softmax_regression.initial_params(X, y)

train_fraction = 0.2

train, test = make_train_and_test_sets(X, y, train_fraction)

iterations = 80
runner_results: list[GradientDescentResult] = [] # Our results

test_set = {
    'w0': start_weight,
    'GD_params': {'lr': [1/100], 'beta': [0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9], 'step_size': None},
    # 'GD_params': {'L': [0.01], 'w0': w0},
    'alg': [Momentum],
    'model': softmax_regression,
    'max_iter': iterations,
    'data_set': train,
    'test_set': test,
    'epsilon':0,
    'auto_stop': False,
    'batch': None
}
runner = Runner(dic = test_set)
runner_results.append(runner.get_result()[0])
runner_results.append(runner.get_result()[1])
runner_results.append(runner.get_result()[2])
runner_results.append(runner.get_result()[3])
runner_results.append(runner.get_result()[4])
runner_results.append(runner.get_result()[5])

#test_set = {
#    'w0': start_weight,
#    'GD_params': {"step_size":1/6},
#    # 'GD_params': {'L': [0.01], 'w0': w0},
#    'alg': [Standard_GD],
#    'model': multinomial_logistic_regression,
#    'max_iter': iterations,
#    'data_set': (feature_array_train, label_array_train),
#    'test_set': (feature_array_test, label_array_test),
#    'epsilon':0,
#    'auto_stop': False,
#    'batch': None
#}
#runner = Runner(dic = test_set)
#runner_results.append(runner.get_result()[0])
#print(ra, runner_results[0].accuracies[0] - runner_results[0].accuracies[iterations-1])

GradientDescentResultPlotter(runner_results).plot_accuracies_over_time().hide_y_axis().legend_placed("center right").with_result_labelled(["Beta = 0", "Beta = 0.1", "Beta = 0.2", "Beta = 0.4", "Beta = 0.8", "Beta = 0.9"]).plot()
