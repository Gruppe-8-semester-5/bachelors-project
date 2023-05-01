import math
import random
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from algorithms.accelerated_GD import Nesterov_acceleration
from datasets.winequality.files import read_wine_data
from analysis.lipschitz import lipschitz_binary_neg_log_likelihood
from datasets.winequality.wine import Wine
from models.logistic_regression import gradient_regularized, predict
from models.logistic_regression import gradient
from algorithms import GradientDescentResult, gradient_descent_template, standard_GD
from algorithms.standard_GD import Standard_GD
from algorithms.momentum_GD import Momentum
from models.utility import accuracy
from analysis.utility import dump_array_to_csv, euclid_distance

# Construct np array of features
def color_to_label(color):
    if color == "white":
        return -1
    return 1

dataset = read_wine_data()
wines: list[Wine] = list(map(lambda d: Wine(d), dataset))
rand_seed = 1
random.seed(rand_seed)
random.shuffle(wines)

from sklearn.preprocessing import StandardScaler

train_fraction = 0.70
data_training_size = math.floor(len(wines) * train_fraction)
train_wines = wines[0:data_training_size]
test_wines = wines[data_training_size:len(wines)]

feature_list_train = list(map(lambda wine: wine.get_feature_vector(), train_wines))
feature_array_train = np.array(feature_list_train)
quality_label_list_train = list(map(lambda wine: wine.get_quality(), train_wines))
quality_label_array_train = np.array(quality_label_list_train)


feature_list_test = list(map(lambda wine: wine.get_feature_vector(), test_wines))
feature_array_test = np.array(feature_list_test)
quality_label_list_test = list(map(lambda wine: wine.get_quality(), test_wines))
quality_label_array_test = np.array(quality_label_list_test)


scaler = StandardScaler()
scaler.fit(feature_array_train)  # Don't cheat - fit only on training data

feature_array_train = scaler.transform(feature_array_train)
feature_array_test = scaler.transform(feature_array_test)

lipschitz = lipschitz_binary_neg_log_likelihood(feature_array_train, quality_label_array_train)
print(f"lipschitz = {lipschitz}")
n = len(quality_label_array_train)
print(f"n = {n}")

from sklearn.linear_model import SGDClassifier

def run_classifier(iter: int) -> SGDClassifier:
    return SGDClassifier(random_state=rand_seed, 
                    loss="log_loss", 
                    alpha=0.05,
                    learning_rate="constant", 
                    early_stopping=False, 
                    fit_intercept=False,
                    n_iter_no_change=20,
                    average=False,
                    eta0=0.0001, 
                    penalty=None, 
                    shuffle=False,
                    tol=np.finfo(np.float64).eps, 
                    epsilon=np.finfo(np.float64).eps,
                    max_iter=iter).fit(feature_array_train, quality_label_array_train)

def run_classifier_regularized(iter: int) -> SGDClassifier:
    return SGDClassifier(random_state=rand_seed, 
                    loss="log_loss", 
                    alpha=0.05,
                    learning_rate="constant", 
                    early_stopping=False, 
                    fit_intercept=False,
                    n_iter_no_change=20,
                    average=False,
                    eta0=0.0001, 
                    penalty="l2", 
                    shuffle=False,
                    tol=np.finfo(np.float64).eps, 
                    epsilon=np.finfo(np.float64).eps,
                    max_iter=iter).fit(feature_array_train, quality_label_array_train)

def score_classifier(classifier: SGDClassifier) -> Tuple[float, float]:
    return (classifier.score(feature_array_train, quality_label_array_train), classifier.score(feature_array_test, quality_label_array_test))

num_iter = run_classifier(2000).n_iter_ # The number of iterations the classifier ran

x_values = [i for i in range(1, num_iter)]

# Calculate difference between E_in and E_out
y_values_in = []
y_values_out = []
y_values_diff = []

y_values_in_l2 = []
y_values_out_l2 = []
y_values_diff_l2 = []


clf = run_classifier(1)
clf_l2 = run_classifier_regularized(1)
for i in range(1, num_iter):
    if i % 10 == 0:
        clf = run_classifier(i)
        clf_l2 = run_classifier_regularized(i)
    in_error, out_error = score_classifier(clf)
    in_error_l2, out_error_l2 = score_classifier(clf_l2)
    y_values_in.append(in_error)
    y_values_out.append(out_error)
    y_values_in_l2.append(in_error_l2)
    y_values_out_l2.append(out_error_l2)

    y_values_diff.append(abs(in_error - out_error))
    y_values_diff_l2.append(abs(in_error_l2 - out_error_l2))


plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.plot(x_values, y_values_diff, label="No regularization")
plt.plot(x_values, y_values_diff_l2, label="L2 regularization")
plt.legend(loc='center right')
plt.show()


plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.plot(x_values, y_values_in, label="No regularization")
plt.plot(x_values, y_values_in_l2, label="L2 regularization")
plt.legend(loc='center right')
plt.show()






raise Exception("\n\nStop here.")




# Construct np array of features
def color_to_label(color):
    if color == "white":
        return 0
    return 1

dataset = read_wine_data()
wines: list[Wine] = list(map(lambda d: Wine(d), dataset))
feature_list = list(map(lambda wine: wine.get_feature_vector(), wines))
feature_array = np.array(feature_list)
color_label_list = list(map(lambda wine: color_to_label(wine.get_color()), wines))
color_label_array = np.array(color_label_list)
lipschitz = lipschitz_binary_neg_log_likelihood(feature_array, color_label_array)
print(f"lipschitz = {lipschitz}")
n = len(dataset)
print(f"n = {n}")

example_wine = wines[0]

feature_size = example_wine.get_feature_vector().size

def make_predictions(weights):
    predictions = []
    for wine in wines:
        predictions.append(predict(weights, wine.get_feature_vector()))
    return predictions

start_gradient = np.zeros(feature_size)
iterations = 299

#Todo: set the best point to one of the tests, so the graph content matches

descent_result_lipchitz: GradientDescentResult = gradient_descent_template.find_minima(
    start_gradient, 
    Standard_GD(1/lipschitz),
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

descent_result_lipchitz_regularized: GradientDescentResult = gradient_descent_template.find_minima(
    start_gradient, 
    Standard_GD(1/lipschitz),
    lambda w: gradient_regularized(feature_array, color_label_array, w, 0.1), 
    max_iter=iterations,
    epsilon=1.0e-3,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

descent_result_lipchitz_regularized.set_most_accurate_weights(descent_result_lipchitz.get_most_accurate_weights())

dataset = read_wine_data()
wines: list[Wine] = list(map(lambda d: Wine(d), dataset))
example_wine = wines[len(wines) - 1].get_feature_vector()

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
x_values = [i for i in range(len(descent_result_lipchitz.get_running_distance_to_most_accurate_weights_average()))]
plt.plot(x_values, descent_result_lipchitz.get_running_distance_to_most_accurate_weights_average(), label="a = 1/L")
plt.plot(x_values, descent_result_lipchitz_regularized.get_running_distance_to_most_accurate_weights_average(), label="a = 1/L (L2)")
plt.legend(loc='upper right')
plt.show()
