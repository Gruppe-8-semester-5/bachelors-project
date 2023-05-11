import matplotlib.pyplot as plt
import numpy as np
from analysis.gradient_descent_result_plotting import GradientDescentResultPlotter
from datasets.winequality.files import read_wine_data
from analysis.lipschitz import lipschitz_binary_neg_log_likelihood
from datasets.winequality.wine import Wine
from models.logistic_regression import negative_log_likelihood, predict
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

def color_to_label(color):
    if color == "white":
        return 0
    return 1

example_wine = wines[0]
print("feature example", example_wine.get_feature_vector())
print("Label example", color_to_label(example_wine.get_color())) 

color_label_list = list(map(lambda wine: color_to_label(wine.get_color()), wines))
color_label_array = np.array(color_label_list)
lipschitz = lipschitz_binary_neg_log_likelihood(feature_array, color_label_array)

feature_size = example_wine.get_feature_vector().size

def make_predictions(weights):
    predictions = []
    for wine in wines:
        predictions.append(predict(weights, wine.get_feature_vector()))

    return predictions

start_gradient = np.random.rand(feature_size)
iterations = 100



#Todo: set the best point to one of the tests, so the graph content matches

descent_result_lipchitz: GradientDescentResult = gradient_descent_template.find_minima(
    start_gradient, 
    Standard_GD(1/lipschitz),
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

first_weight = descent_result_lipchitz.get_first_weight()
final_weight = descent_result_lipchitz.get_final_weight()
normDiff = np.linalg.norm(first_weight - final_weight)
expected_rate_function = lambda x : ((lipschitz * normDiff**2)/2) * (1/x if x>0 else 1)

log_likelihood_values = lambda x : negative_log_likelihood(feature_array, color_label_array,descent_result_lipchitz.get_weights_over_time()[x])

GradientDescentResultPlotter([descent_result_lipchitz]).plot_best_weight_distance_to_zero_gradient_over_time().plot_function(log_likelihood_values).legend_placed("upper right").with_result_labelled(["GD"]).with_functions_labelled(["1/x*700k", "derived rate"]).plot()
