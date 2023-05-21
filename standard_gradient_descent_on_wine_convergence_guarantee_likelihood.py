import matplotlib.pyplot as plt
import numpy as np
from algorithms.accelerated_GD import Nesterov_acceleration
from analysis.gradient_descent_result_plotting import GradientDescentResultPlotter
from datasets.winequality.files import read_wine_data
from analysis.lipschitz import lipschitz_binary_neg_log_likelihood
from datasets.winequality.wine import Wine
from models.logistic_regression import gradient, negative_log_likelihood, predict
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
iterations = 100

descent_result_lipchitz_with_w_star: GradientDescentResult = gradient_descent_template.find_minima(
    start_gradient, 
    Standard_GD(1/(lipschitz)),
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=100000,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

descent_result_lipchitz: GradientDescentResult = gradient_descent_template.find_minima(
    start_gradient, 
    Standard_GD(1/(lipschitz)),
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

weights = descent_result_lipchitz.get_weights_over_time()
best_nll = negative_log_likelihood(feature_array, color_label_array, weights[0])

likelihood_values = [negative_log_likelihood(feature_array, color_label_array, weight) for weight in weights]


#likelihood_values = []
#for weight in weights:
#    current_nll = negative_log_likelihood(feature_array, color_label_array, weight)
#    if current_nll > best_nll:
#        best_nll = current_nll
#    likelihood_values.append(best_nll)


f_w_star = negative_log_likelihood(feature_array, color_label_array, descent_result_lipchitz_with_w_star.get_best_weights_over_time()[len(descent_result_lipchitz_with_w_star.get_best_weights_over_time())-1])

diff_to_w_star = [np.abs(nll_val - f_w_star) for nll_val in likelihood_values]
best_diff_to_w_star = diff_to_w_star[0]
distances_to_w_star_only_best = [ ]

for current_diff_to_w_star in diff_to_w_star:
    best_diff_to_w_star = min(best_diff_to_w_star, current_diff_to_w_star)
    distances_to_w_star_only_best.append(best_diff_to_w_star)

convergence_rate_multiplier = 8000 # Labeled C in plot
GradientDescentResultPlotter([descent_result_lipchitz]).plot_function(lambda x: distances_to_w_star_only_best[x]).legend_placed("upper right").with_functions_labelled(["f(w_x) - f(w*)", "C/x"]).with_x_values(range(0,iterations)).plot()

#GradientDescentResultPlotter([descent_result_lipchitz]).plot_function(lambda x :-likelihood_values[x] if -likelihood_values[x] > 0 else 0).plot_function(lambda x: convergence_rate_multiplier/x if x != 0 else convergence_rate_multiplier).legend_placed("upper right").with_functions_labelled(["f(w)", "2.750.000/x"]).with_x_values(range(0,iterations)).plot()


#.set_y_limit(0, 0.016*1e6)