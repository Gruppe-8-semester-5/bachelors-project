import math
import random
import matplotlib.pyplot as plt
import numpy as np
from algorithms.accelerated_GD import Nesterov_acceleration
from datasets.winequality.files import read_wine_data
from analysis.lipschitz import lipschitz_binary_neg_log_likelihood
from datasets.winequality.wine import Wine
from models.logistic_regression import predict
from models.logistic_regression import gradient
from algorithms import GradientDescentResult, gradient_descent_template, standard_GD
from algorithms.standard_GD import Standard_GD
from algorithms.momentum_GD import Momentum
from models.utility import accuracy
from analysis.utility import dump_array_to_csv, euclid_distance

# Construct np array of features
def quality_to_label(quality):
    result = (quality - 3) / 6
    print(result)
    return result

dataset = read_wine_data()
wines: list[Wine] = list(map(lambda d: Wine(d), dataset))

# We specify seed to make multiple runs the same
random.Random(741238).shuffle(wines)

no_of_training_samples = 4000
train_wines = wines[0:no_of_training_samples]
print(f"Training on {len(train_wines)}")
test_wines = wines[no_of_training_samples:len(wines)-1]
print(f"Testing on {len(test_wines)}")

feature_length = wines[0].get_feature_vector().size

train_feature_list = list(map(lambda wine: wine.get_feature_vector(), train_wines))
train_feature_array = np.array(train_feature_list)
train_quality_label_list = list(map(lambda wine: quality_to_label(wine.get_quality()), train_wines))
train_quality_label_array = np.array(train_quality_label_list)
lipschitz = lipschitz_binary_neg_log_likelihood(train_feature_array, train_quality_label_array)

test_feature_list = list(map(lambda wine: wine.get_feature_vector(), test_wines))
test_feature_array = np.array(test_feature_list)
test_quality_label_list = list(map(lambda wine: quality_to_label(wine.get_quality()), test_wines))
test_quality_label_array = np.array(test_quality_label_list)
test_lipschitz = lipschitz_binary_neg_log_likelihood(test_feature_array, test_quality_label_array)


def make_predictions(weights):
    predictions = []
    for wine in test_wines:
        predictions.append(predict(weights, wine.get_feature_vector()))
    return predictions

start_gradient = np.zeros(feature_length)
iterations = 500

gd_result: GradientDescentResult = gradient_descent_template.find_minima(
    start_gradient, 
    Standard_GD(0.01),
    lambda w: gradient(train_feature_array, train_quality_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-99,
    accuracy=(lambda w: accuracy(test_quality_label_array, make_predictions(w))))

# plots distance to best lipschitz point
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
x_values = [i for i in range(len(gd_result.get_accuracy_over_time()))]
plt.plot(x_values, gd_result.get_accuracy_over_time(), label="a = 0.01")
plt.legend(loc='upper right')
plt.show()

print(gd_result.get_accuracy_over_time()[len(gd_result.get_accuracy_over_time())-1])