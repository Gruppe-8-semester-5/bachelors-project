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
iterations = 500


descent_result_lipchitz: GradientDescentResult = gradient_descent_template.find_minima(
    start_gradient, 
    Standard_GD(1/lipschitz),
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))



# Notes for nestorov:
# - Lower alphas seem to be best (< 0.01)

descent_result_nestorov1: GradientDescentResult = gradient_descent_template.find_minima(
    start_gradient, 
    Nesterov_acceleration(np.zeros(feature_size), 1/lipschitz, 0.01, 0.01, 0.01),
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

descent_result_nestorov2: GradientDescentResult = gradient_descent_template.find_minima(
    start_gradient, 
    Nesterov_acceleration(np.zeros(feature_size), 1/lipschitz, 0.1, 0.005, 0.01),
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

descent_result_nestorov3: GradientDescentResult = gradient_descent_template.find_minima(
    start_gradient, 
    Nesterov_acceleration(np.zeros(feature_size), 1/lipschitz, 0.1, 0.001, 0.01),
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

# descent_result_nestorov4: GradientDescentResult = gradient_descent_template.find_minima(
#     start_gradient, 
#     Nesterov_acceleration(np.zeros(feature_size), 0.2, 0.01, 0.1, 0.01),
#     lambda w: gradient(feature_array, color_label_array, w), 
#     max_iter=iterations,
#     epsilon=1.0e-2,
#     accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

descent_result_momentum: GradientDescentResult = gradient_descent_template.find_minima(
    start_gradient, 
    Momentum(),
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

best_lipschitz_point = descent_result_lipchitz.get_most_accurate_weights()

items_to_compare = [
    descent_result_lipchitz,
    descent_result_nestorov1,
    descent_result_nestorov2,
    descent_result_nestorov3,
    descent_result_momentum
]

best_performer = descent_result_lipchitz
i = 0
for result in items_to_compare:
    if result.get_best_accuracy() < best_performer.get_best_accuracy():
        best_performer = result
        print(f"best_performer = {i}")
    i+=1

for result in items_to_compare:
    result.set_most_accurate_weights(best_performer.get_most_accurate_weights())

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
x_values = [i for i in range(len(descent_result_lipchitz.get_distances_to_most_accurate_weight()))]
plt.plot(x_values, descent_result_lipchitz.get_distances_to_most_accurate_weight(), label="No acceleration")
plt.plot(x_values, descent_result_nestorov1.get_distances_to_most_accurate_weight(), label="Nestorov1")
plt.plot(x_values, descent_result_nestorov2.get_distances_to_most_accurate_weight(), label="Nestorov2")
plt.plot(x_values, descent_result_nestorov3.get_distances_to_most_accurate_weight(), label="Nestorov3")
plt.plot(x_values, descent_result_momentum.get_distances_to_most_accurate_weight(), label="Momentum")
plt.legend(loc='upper left')
plt.show()
