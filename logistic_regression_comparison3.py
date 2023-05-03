import math
import random
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from algorithms.accelerated_GD import Nesterov_acceleration
from datasets.winequality.files import read_wine_data
from analysis.lipschitz import lipschitz_binary_neg_log_likelihood
from datasets.winequality.wine import Wine
from logistic_regression_comparison import make_predictions
from models.logistic_regression import gradient_regularized, predict, predict_with_softmax
from models.logistic_regression import gradient
from algorithms import GradientDescentResult, gradient_descent_template, standard_GD
from algorithms.standard_GD import Standard_GD
from algorithms.momentum_GD import Momentum
from models.utility import accuracy, accuracy_k_encoded
from analysis.utility import dump_array_to_csv, euclid_distance

dataset = read_wine_data()
wines: list[Wine] = list(map(lambda d: Wine(d), dataset))
rand_seed = 1
random.seed(rand_seed)
random.shuffle(wines)

K = 7

def K_encoded(label):
    """Turns a 1x1 label into a 1xK label array, with all 0s and a single 1"""
    encoded = np.zeros(K)
    encoded[label - 3] = 1
    return encoded

train_fraction = 0.70
data_training_size = math.floor(len(wines) * train_fraction)
train_wines = wines[0:data_training_size]
test_wines = wines[data_training_size:len(wines)]

example_wine = wines[0]
feature_size = example_wine.get_feature_vector().size
start_weight = np.ones((feature_size, K))

feature_list_train = list(map(lambda wine: wine.get_feature_vector(), train_wines))
feature_array_train = np.array(feature_list_train)
feature_list_test = list(map(lambda wine: wine.get_feature_vector(), test_wines))
feature_array_test = np.array(feature_list_test)

label_list_train= list(map(lambda wine: wine.get_quality(), train_wines))
label_array_train = np.array(label_list_train)


label_list_train_k = list(map(lambda wine: K_encoded(wine.get_quality()), train_wines))
label_array_train_k = np.array(label_list_train)

label_list_test = list(map(lambda wine: wine.get_quality(), test_wines))
label_array_test = np.array(label_list_test)

label_list_test_k = list(map(lambda wine: K_encoded(wine.get_quality()), test_wines))
label_array_test_k = np.array(label_list_test)


iterations = 1000

print (predict_with_softmax(start_weight, example_wine.get_feature_vector()))

def make_predictions(weights, wine_set):
    predictions = []
    for wine in wine_set:
        pred = predict_with_softmax(weights, wine.get_feature_vector())
        predictions.append(pred)
    return np.array(predictions)

descent_result_lipchitz: GradientDescentResult = gradient_descent_template.find_minima(
    start_weight, 
    Standard_GD(0.1),
    lambda w: gradient(feature_array_train, label_array_train, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy_k_encoded(label_array_train, make_predictions(w, train_wines))))

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
x_values = [x for x in range(0, len(descent_result_lipchitz.get_accuracy_over_time()))]
y_values = descent_result_lipchitz.get_accuracy_over_time()
plt.plot(x_values, y_values)
plt.legend(loc='center right')
plt.show()
