import math
import random
import matplotlib.pyplot as plt
import numpy as np
from analysis.gradient_descent_result_plotting import GradientDescentResultPlotter
from datasets.winequality.files import read_wine_data
from datasets.winequality.wine import Wine
from models.logistic_regression import gradient_k, predict_with_softmax
from algorithms import GradientDescentResult, gradient_descent_template
from algorithms.standard_GD import Standard_GD
from algorithms.momentum_GD import Momentum
from algorithms.accelerated_GD import Nesterov_acceleration
from models.utility import accuracy_k_encoded
from models import multinomial_logistic_regression
from test_runner.test_runner_file import Runner

#for ra in range(0, 100):

dataset = read_wine_data()
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

example_wine = wines[0]
feature_size = example_wine.get_feature_vector().size

start_weight = np.random.normal(size=(feature_size, K))

feature_list_train = list(map(lambda wine: wine.get_feature_vector(), train_wines))
feature_array_train = np.array(feature_list_train)
feature_list_test = list(map(lambda wine: wine.get_feature_vector(), test_wines))
feature_array_test = np.array(feature_list_test)

label_list_train= list(map(lambda wine: wine.get_quality(), train_wines))
label_array_train = np.array(label_list_train)
label_list_test = list(map(lambda wine: wine.get_quality(), test_wines))
label_array_test = np.array(label_list_test)

iterations = 80
runner_results: list[GradientDescentResult] = [] # Our results
#test_set = {
#    'w0': start_weight,
#    'GD_params': {"w0": start_weight, 'alpha': [1/60.2], "mu": [0.01], "L":[0.01], "beta": [0.01]},
#    # 'GD_params': {'L': [0.01], 'w0': w0},
#    'alg': [Nesterov_acceleration],
#    'model': multinomial_logistic_regression,
#    'max_iter': iterations,
#    'data_set': (feature_array_train, label_array_train),
#    'test_set': (feature_array_test, label_array_test),
#    'epsilon':0,
#    'auto_stop': False,
#    'batch': None
#}
#runner = Runner(dic = test_set)
#runner_results: list[GradientDescentResult] = runner.get_result()

test_set = {
    'w0': start_weight,
    'GD_params': {'lr': [1/100], 'beta': [0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9], 'step_size': None},
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

GradientDescentResultPlotter(runner_results).plot_accuracies_over_time().hide_y_axis().legend_placed("center right").with_labels(["Beta = 0", "Beta = 0.1", "Beta = 0.2", "Beta = 0.4", "Beta = 0.8", "Beta = 0.9"]).plot()

raise Exception("Stop prematurely")

#simple_result: GradientDescentResult = runner.get_result(dic = test_set | {'alg': Standard_GD})[0]
nest_result: GradientDescentResult = runner.get_result()[0]
print(nest_result.weights)
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
x_values = [x for x in range(0, len(nest_result.get_derivation_distances_to_zero_over_time()))]
y_values_nest = nest_result.get_derivation_distances_to_zero_over_time()
#y_values_simple = simple_result.get_derivation_distances_to_zero_over_time()
#plt.plot(x_values, y_values_simple, label="Simple")
plt.plot(x_values, y_values_nest, label="Nesty")
print("plotting")
plt.legend(loc='center right')
plt.show()


descent_result_lipchitz: GradientDescentResult = gradient_descent_template.find_minima(
    start_weight, 
    Standard_GD(0.1),
    lambda w: gradient_k(feature_array_train, label_array_train, w), 
    max_iter=iterations,
    epsilon=1.0e-1000,
    accuracy=(lambda w: accuracy_k_encoded(label_array_train, make_predictions(w, train_wines))))


