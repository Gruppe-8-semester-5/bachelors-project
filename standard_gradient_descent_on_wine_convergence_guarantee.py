import matplotlib.pyplot as plt
import numpy as np
from algorithms.accelerated_GD import Nesterov_acceleration
from analysis.gradient_descent_result_plotting import GradientDescentResultPlotter
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
        return -1.
    return 1.

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
# print("feature example", example_wine.get_feature_vector())
# print("Label example", color_to_label(example_wine.get_color())) 

feature_size = example_wine.get_feature_vector().size

def make_predictions(weights):
    predictions = []
    for wine in wines:
        predictions.append(predict(weights, wine.get_feature_vector()))
    return predictions

start_gradient = np.zeros(feature_size)
iterations = 1000


descent_result_lipchitz: GradientDescentResult = gradient_descent_template.find_minima(
    start_gradient, 
    Standard_GD(1/(lipschitz)),
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

descent_result_025lipchitz: GradientDescentResult = gradient_descent_template.find_minima(
    start_gradient, 
    Standard_GD(1/(lipschitz/4)),
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

descent_result_05lipchitz: GradientDescentResult = gradient_descent_template.find_minima(
    start_gradient, 
    Standard_GD(1/(lipschitz/2)),
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

descent_result_2lipchitz: GradientDescentResult = gradient_descent_template.find_minima(
    start_gradient, 
    Standard_GD(1/(lipschitz*2)),
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

descent_result_4lipchitz: GradientDescentResult = gradient_descent_template.find_minima(
    start_gradient, 
    Standard_GD(1/(lipschitz*4)),
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

# Plot best weight to absolute best weight
# Figure out which performed the best
results: list[GradientDescentResult] = [
    descent_result_025lipchitz, 
    descent_result_05lipchitz, 
    descent_result_lipchitz, 
    descent_result_2lipchitz, 
    descent_result_4lipchitz
]

closest_distance = np.Infinity
best_performer = None
for result in results:
    result_closest = result.get_best_weight_derivation_distances_to_zero_over_time()[iterations-1]
    if result_closest < closest_distance:
        closest_distance = result_closest
        best_performer = result

## Then we set the best weights of the result, such that the performance calculations will be relative to the best result
for result in results:
    result.set_closest_to_zero_derivation_weight(best_performer.get_closest_to_zero_derivation_weights_over_time()[iterations - 1])

convergence_rate_multiplier = 4000000 # Labeled C in plot

GradientDescentResultPlotter(results).plot_best_weight_distance_to_zero_gradient_over_time(1).set_y_limit(0, 0.1*1e6).plot_function(lambda x: convergence_rate_multiplier/x if x != 0 else convergence_rate_multiplier).with_result_labelled(["1/0.25L","1/0.5L","1/L","1/2L","1/4L"]).with_functions_labelled(["C/x"]).legend_placed("upper left").plot()