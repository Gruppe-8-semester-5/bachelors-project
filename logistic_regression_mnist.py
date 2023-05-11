import math
import random
import matplotlib.pyplot as plt
import numpy as np
from analysis.gradient_descent_result_plotting import GradientDescentResultPlotter
from datasets.mnist.files import read_train_data, read_test_data
from datasets.winequality.wine import Wine
from models.logistic_regression import gradient_k, predict_with_softmax
from algorithms import GradientDescentResult, gradient_descent_template
from algorithms.standard_GD import Standard_GD
from algorithms.momentum_GD import Momentum
from algorithms.accelerated_GD import Nesterov_acceleration
from models.utility import accuracy_k_encoded
from models import multinomial_logistic_regression
from test_runner.test_runner_file import Runner
from sklearn.preprocessing import PolynomialFeatures
#for ra in range(0, 100):

X_train = read_train_data()

print(X_train)


exit()

wines: list[Wine] = list(map(lambda d: Wine(d), dataset))
rand_seed = 33
random.seed(rand_seed)
random.shuffle(wines)
np.random.seed(rand_seed)
K = 7

train_fraction = 0.50
data_training_size = math.floor(len(wines) * train_fraction)
train_wines = wines[0:data_training_size]
test_wines = wines[data_training_size:len(wines)]

poly_features = PolynomialFeatures(degree=2, include_bias=False)

example_wine = wines[0]
feature_size = example_wine.get_feature_vector().size

start_weight = np.random.normal(size=(feature_size, K))

feature_list_train = list(map(lambda wine: wine.get_feature_vector(), train_wines))
feature_array_train = poly_features.fit_transform(np.array(feature_list_train).reshape(-1,1))
feature_list_test = list(map(lambda wine: wine.get_feature_vector(), test_wines))
feature_array_test = poly_features.fit_transform(np.array(feature_list_test).reshape(-1,1))

label_list_train= list(map(lambda wine: wine.get_quality(), train_wines))
label_array_train = np.array(label_list_train)
label_list_test = list(map(lambda wine: wine.get_quality(), test_wines))
label_array_test = np.array(label_list_test)

# Compare different values for beta in nesterov
iterations = 50
runner_results = [] # Our results
test_set = {
    'w0': start_weight,
    'GD_params': {'w0': start_weight, 'L':None,'mu':None, 'alpha': [0.01], 'beta': [0.9,0.7,0.5,0.1]},
    # 'GD_params': {'L': [0.01], 'w0': w0},
    'alg': [Nesterov_acceleration],
    'model': multinomial_logistic_regression,
    'max_iter': iterations,
    'data_set': (feature_array_train, label_array_train),
    'test_set': (feature_array_test, label_array_test),
    'epsilon':0,
    'auto_stop': False,
    'batch': None
}
runner = Runner(dic = test_set)
runner_results.append(runner.get_result()[0])
runner_results.append(runner.get_result()[1])
runner_results.append(runner.get_result()[2])
runner_results.append(runner.get_result()[3])
GradientDescentResultPlotter(runner_results).plot_distance_to_zero_gradient_over_time().hide_y_axis().legend_placed("center right").with_result_labelled(["beta = 0.9","beta = 0.7","beta = 0.5","beta = 0.1"]).plot()



# Compare nesterov and momentum
iterations = 20
runner_results = [] # Our results
test_set = {
    'w0': start_weight,
    'GD_params': {'lr': [0.05], 'beta': [0.9]},
    # 'GD_params': {'L': [0.01], 'w0': w0},
    'alg': [Momentum],
    'model': multinomial_logistic_regression,
    'max_iter': iterations,
    'data_set': (feature_array_train, label_array_train),
    'test_set': (feature_array_test, label_array_test),
    'epsilon':0,
    'auto_stop': False,
    'batch': None
}
runner = Runner(dic = test_set)
runner_results.append(runner.get_result()[0])
test_set = {
    'w0': start_weight,
    'GD_params': {'w0': start_weight, 'L':None,'mu':None, 'alpha': [0.05], 'beta': [0.9]},
    # 'GD_params': {'L': [0.01], 'w0': w0},
    'alg': [Nesterov_acceleration],
    'model': multinomial_logistic_regression,
    'max_iter': iterations,
    'data_set': (feature_array_train, label_array_train),
    'test_set': (feature_array_test, label_array_test),
    'epsilon':0,
    'auto_stop': False,
    'batch': None
}
runner = Runner(dic = test_set)
runner_results.append(runner.get_result()[0])
test_set = {
    'w0': start_weight,
    'GD_params': {'step_size': [0.05]},
    # 'GD_params': {'L': [0.01], 'w0': w0},
    'alg': [Standard_GD],
    'model': multinomial_logistic_regression,
    'max_iter': iterations,
    'data_set': (feature_array_train, label_array_train),
    'test_set': (feature_array_test, label_array_test),
    'epsilon':0,
    'auto_stop': False,
    'batch': None
}
runner = Runner(dic = test_set)
runner_results.append(runner.get_result()[0])
GradientDescentResultPlotter(runner_results).plot_accuracies_over_time().legend_placed("center right").with_result_labelled(["Momentum", "NAG", "No acceleration"]).plot()
