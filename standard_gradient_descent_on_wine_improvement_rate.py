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
dataset = read_wine_data()
wines: list[Wine] = list(map(lambda d: Wine(d), dataset))
feature_list = list(map(lambda wine: wine.get_feature_vector(), wines))
feature_array = np.array(feature_list)
lipschitz = lipschitz_binary_neg_log_likelihood(feature_array)
print(f"lipschitz = {lipschitz}")
n = len(dataset)
print(f"n = {n}")


def color_to_label(color):
    if color == "white":
        return 0
    return 1

example_wine = wines[0]

color_label_list = list(map(lambda wine: color_to_label(wine.get_color()), wines))
color_label_array = np.array(color_label_list)

feature_size = example_wine.get_feature_vector().size

def make_predictions(weights):
    predictions = []
    for wine in wines:
        predictions.append(predict(weights, wine.get_feature_vector()))
    return predictions

start_gradient = np.zeros(feature_size)
iterations = 100000

descent_result_lipchitz: GradientDescentResult = gradient_descent_template.find_minima(
    start_gradient, 
    Standard_GD(1/lipschitz),
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

dataset = read_wine_data()
wines: list[Wine] = list(map(lambda d: Wine(d), dataset))
example_wine = wines[len(wines) - 1].get_feature_vector()
print(wines[len(wines)-1].get_quality())


# plots distance to best lipschitz point

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
x_values = [i for i in range(len(descent_result_lipchitz.get_distances_to_most_accurate_weight()))]
plt.plot(x_values, descent_result_lipchitz.get_distances_to_most_accurate_weight(), label="a = 1/L")
plt.legend(loc='upper right')
plt.show()


dump_array_to_csv(descent_result_lipchitz.get_distances_to_most_accurate_weight(), "improvement_deltas_1_over_lipschitz.csv", True, "100k")
