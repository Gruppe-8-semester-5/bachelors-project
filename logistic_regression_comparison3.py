import math
import random
import matplotlib.pyplot as plt
import numpy as np
from datasets.winequality.files import read_wine_data
from datasets.winequality.wine import Wine
from models.logistic_regression import gradient_k, predict_with_softmax
from algorithms import GradientDescentResult, gradient_descent_template
from algorithms.standard_GD import Standard_GD
from algorithms.accelerated_GD import Nesterov_acceleration
from models.utility import accuracy_k_encoded
from models import multinomial_logistic_regression
from test_runner.test_runner_file import Runner

dataset = read_wine_data()
wines: list[Wine] = list(map(lambda d: Wine(d), dataset))
rand_seed = 1
random.seed(rand_seed)
random.shuffle(wines)

K = 7

train_fraction = 0.10
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

iterations = 1000

test_set = {
    'w0': start_weight,
    'GD_params': {"w0": start_weight, 'step_size': [0.0000001]},
    # 'GD_params': {'L': [0.01], 'w0': w0},
    'alg': [Nesterov_acceleration],
    'model': multinomial_logistic_regression,
    'max_iter': iterations,
    'data_set': (feature_array_train, label_array_train),
    'test_set': (feature_array_test, label_array_test),
    'epsilon':0,
    'batch': None
}

runner = Runner(dic = test_set)
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

raise Exception("Stop prematurely")

descent_result_lipchitz: GradientDescentResult = gradient_descent_template.find_minima(
    start_weight, 
    Standard_GD(0.1),
    lambda w: gradient_k(feature_array_train, label_array_train, w), 
    max_iter=iterations,
    epsilon=1.0e-1000,
    accuracy=(lambda w: accuracy_k_encoded(label_array_train, make_predictions(w, train_wines))))


