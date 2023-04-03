import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Process
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

# Maps colors to 0 or 1
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
iterations = 1000

# In order to run useful experiments, test all of the following values
testable_mus = [0.005, 0.01, 0.05, 0.1, 0.5]
testable_Ls = [0.005, 0.01, 0.05, 0.1, 0.5]
testable_alphas = [0.005, 0.01, 0.05, 0.1, 0.5, 1]
testable_betas = [0.005, 0.01, 0.05, 0.1, 0.5, 1]
for mu in testable_mus:
    for L in testable_Ls:
        for alpha in testable_alphas:
            for beta in testable_betas:
                print(mu,L,alpha,beta)
                result = gradient_descent_template.find_minima(
                    start_gradient, 
                    Nesterov_acceleration(start_gradient, L, mu, alpha, beta),
                    lambda w: gradient(feature_array, color_label_array, w), 
                    max_iter=iterations,
                    epsilon=1.0e-2,
                    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))
                csv_name = f"nestorov_L{L}_mu{mu}_alpha{alpha}_beta{beta}.csv"
                dump_array_to_csv(result.get_accuracy_over_time(), f"accuracy_over_time_{csv_name}", override=True)
                dump_array_to_csv(result.get_best_weight_over_time_distances_to_best_weight(), f"best_weight_over_time_distances_to_best_weight_{csv_name}", override=True)
                dump_array_to_csv(result.get_best_weights_over_time(), f"best_weights_over_time_{csv_name}", override=True)
                dump_array_to_csv(result.get_distances_to_best_weight(), f"distances_to_best_weight_{csv_name}", override=True)
                dump_array_to_csv(result.get_distances_to_final_weight(), f"distances_to_final_weight_{csv_name}", override=True)
                dump_array_to_csv(result.get_running_accuracy_average(), f"running_accuracy_average_{csv_name}", override=True)
                dump_array_to_csv(result.get_running_distance_to_best_weights_average(), f"running_distance_to_best_weights_average_{csv_name}", override=True)
