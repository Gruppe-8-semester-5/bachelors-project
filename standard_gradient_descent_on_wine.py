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
# print("feature example", example_wine.get_feature_vector())
# print("Label example", color_to_label(example_wine.get_color())) 

feature_size = example_wine.get_feature_vector().size

def make_predictions(weights):
    predictions = []
    for wine in wines:
        predictions.append(predict(weights, wine.get_feature_vector()))
    return predictions

start_gradient = np.zeros(feature_size)
iterations = 100000

#Todo: set the best point to one of the tests, so the graph content matches

descent_result_lipchitz: GradientDescentResult = gradient_descent_template.find_minima(
    start_gradient, 
    Standard_GD(1/lipschitz),
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))
descent_result_l50: GradientDescentResult = gradient_descent_template.find_minima(
    start_gradient, 
    Standard_GD(1/5),
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))
descent_result_l40: GradientDescentResult = gradient_descent_template.find_minima(
    start_gradient, 
    Standard_GD(1/4),
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))
descent_result_l70: GradientDescentResult = gradient_descent_template.find_minima(
    start_gradient, 
    Standard_GD(1/7),
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))
descent_result_l80: GradientDescentResult = gradient_descent_template.find_minima(
    start_gradient, 
    Standard_GD(1/8),
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))
# descent_result_07: GradientDescentResult = gradient_descent_template.find_minima(
#     start_gradient, 
#     Standard_GD(0.7),
#     lambda w: gradient(feature_array, color_label_array, w), 
#     max_iter=iterations,
#     epsilon=1.0e-2,
#     accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))
# descent_result_05: GradientDescentResult = gradient_descent_template.find_minima(
#     start_gradient, 
#     Standard_GD(0.5),
#     lambda w: gradient(feature_array, color_label_array, w), 
#     max_iter=iterations,
#     epsilon=1.0e-2,
#     accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

#descent_result_0001: GradientDescentResult = gradient_descent_template.find_minima(
#    start_gradient, 
#    Standard_GD(0.001),
#    lambda w: gradient(feature_array, color_label_array, w), 
#    max_iter=iterations,
#    epsilon=1.0e-2,
#    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))
#
#descent_result_001: GradientDescentResult = gradient_descent_template.find_minima(
#    start_gradient, 
#    Standard_GD(0.01),
#    lambda w: gradient(feature_array, color_label_array, w), 
#    max_iter=iterations,
#    epsilon=1.0e-2,
#    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))
#
#descent_result_005: GradientDescentResult = gradient_descent_template.find_minima(
#    start_gradient, 
#    Standard_GD(0.05),
#    lambda w: gradient(feature_array, color_label_array, w), 
#    max_iter=iterations,
#    epsilon=1.0e-2,
#    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))
#
#descent_result_01: GradientDescentResult = gradient_descent_template.find_minima(
#    start_gradient, 
#    Standard_GD(0.1),
#    lambda w: gradient(feature_array, color_label_array, w), 
#    max_iter=iterations,
#    epsilon=1.0e-2,
#    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))
#
#descent_result_05: GradientDescentResult = gradient_descent_template.find_minima(
#    start_gradient,
#    Standard_GD(0.5),
#    lambda w: gradient(feature_array, color_label_array, w), 
#    max_iter=iterations,
#    epsilon=1.0e-2,
#    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))
#
#descent_result_1: GradientDescentResult = gradient_descent_template.find_minima(
#    start_gradient, 
#    Standard_GD(1),
#    lambda w: gradient(feature_array, color_label_array, w), 
#    max_iter=iterations,
#    epsilon=1.0e-2,
#    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

dataset = read_wine_data()
wines: list[Wine] = list(map(lambda d: Wine(d), dataset))
example_wine = wines[len(wines) - 1].get_feature_vector()
print(wines[len(wines)-1].get_quality())


# plots distance to final point
# number_of_points = descent_result.number_of_points()
# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True
# x_values = [i for i in range(number_of_points)]
# y_values = descent_result.get_distances_to_final_point()
# plt.plot(x_values, y_values, linestyle="-")
# plt.show()


# Plot accuracy over time
#number_of_points = descent_result_lipchitz.number_of_points()
#plt.rcParams["figure.figsize"] = [7.50, 3.50]
#plt.rcParams["figure.autolayout"] = True
#x_values = [i for i in range(number_of_points)]
#plt.plot(x_values, descent_result_lipchitz.get_accuracy_over_time(), label="a = 1/L")
#plt.plot(x_values, descent_result_0001.get_accuracy_over_time(), label="a = 0.001")
#plt.plot(x_values, descent_result_001.get_accuracy_over_time(), label="a = 0.01")
#plt.plot(x_values, descent_result_005.get_accuracy_over_time(), label="a = 0.05")
#plt.plot(x_values, descent_result_01.get_accuracy_over_time(), label="a = 0.1")
#plt.plot(x_values, descent_result_05.get_accuracy_over_time(), label="a = 0.5")
#plt.plot(x_values, descent_result_1.get_accuracy_over_time(), label="a = 1")
#plt.show()


# plots distance to best point
# number_of_points = descent_result_lipchitz.number_of_points()
# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True
# x_values = [i for i in range(number_of_points)]
# y_values = descent_result_lipchitz.get_distances_to_best_point()
# plt.plot(x_values, y_values, label="a = 1/L", linestyle="-")
# plt.plot(x_values, descent_result_0001.get_distances_to_best_point(), label="a = 0.01")
# plt.plot(x_values, descent_result_001.get_distances_to_best_point(), label="a = 0.01")
# plt.plot(x_values, descent_result_005.get_distances_to_best_point(), label="a = 0.05")
# plt.plot(x_values, descent_result_01.get_distances_to_best_point(), label="a = 0.1")
# plt.plot(x_values, descent_result_05.get_distances_to_best_point(), label="a = 0.5")
# plt.plot(x_values, descent_result_1.get_distances_to_best_point(), label="a = 1")
# plt.legend(loc='upper right')
# plt.show()


# plots distance to best lipschitz point
best_lipschitz_point = descent_result_lipchitz.get_most_accurate_weights()

# descent_result_0001.set_best_weights(best_lipschitz_point)
# descent_result_001.set_best_weights(best_lipschitz_point)
# descent_result_005.set_best_weights(best_lipschitz_point)
# descent_result_01.set_best_weights(best_lipschitz_point)
# descent_result_05.set_best_weights(best_lipschitz_point)
# descent_result_1.set_best_weights(best_lipschitz_point)

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True
# x_values = [i for i in range(number_of_points)]
# y_values = descent_result_lipchitz.get_distances_to_best_point()
# plt.plot(x_values, y_values, label="a = 1/L", linestyle="-")
# plt.plot(x_values, descent_result_0001.get_distances_to_best_point(), label="a = 0.01")
# plt.plot(x_values, descent_result_001.get_distances_to_best_point(), label="a = 0.01")
# plt.plot(x_values, descent_result_005.get_distances_to_best_point(), label="a = 0.05")
# plt.plot(x_values, descent_result_01.get_distances_to_best_point(), label="a = 0.1")
# plt.plot(x_values, descent_result_05.get_distances_to_best_point(), label="a = 0.5")
# plt.plot(x_values, descent_result_1.get_distances_to_best_point(), label="a = 1")
# plt.legend(loc='upper right')
# plt.show()

# dump_array_to_csv(descent_result_lipchitz.get_distances_to_best_point(), "lipschitz_dist_to_best_gradients.csv")

# Plot best weight to respective best weight
# number_of_points = descent_result_lipchitz.number_of_points()
# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True
# x_values = [i for i in range(len(descent_result_lipchitz.get_best_weight_over_time_distances_to_best_weight()))]
# plt.plot(x_values, descent_result_lipchitz.get_best_weight_over_time_distances_to_best_weight(), label="a = 1/L")
# plt.plot(x_values, descent_result_0001.get_best_weight_over_time_distances_to_best_weight(), label="a = 0.001")
# plt.plot(x_values, descent_result_001.get_best_weight_over_time_distances_to_best_weight(), label="a = 0.01")
# plt.plot(x_values, descent_result_005.get_best_weight_over_time_distances_to_best_weight(), label="a = 0.05")
# plt.plot(x_values, descent_result_01.get_best_weight_over_time_distances_to_best_weight(), label="a = 0.1")
# #plt.plot(x_values, descent_result_05.get_best_weight_over_time_distances_to_best_weight(), label="a = 0.5")
# #plt.plot(x_values, descent_result_1.get_best_weight_over_time_distances_to_best_weight(), label="a = 1")
# plt.legend(loc='upper right')
# plt.show()


# Plot best weight to absolute best weight
# Figure out which performed the best
to_compare: list[GradientDescentResult] = [
    descent_result_lipchitz,
    descent_result_l40,
    descent_result_l50,
    descent_result_l70,
    descent_result_l80,
]
best_accuracy = -1
best_performer = None
for result in to_compare:
    if result.get_best_accuracy() > best_accuracy:
        best_accuracy = result.get_best_accuracy()
        best_performer = result

## Then we set the best weights of the result, such that the performance calculations will be relative to the best result
descent_result_lipchitz.set_most_accurate_weights(best_performer.get_final_weight())
descent_result_l40.set_most_accurate_weights(best_performer.get_final_weight())
descent_result_l50.set_most_accurate_weights(best_performer.get_final_weight())
descent_result_l70.set_most_accurate_weights(best_performer.get_final_weight())
descent_result_l80.set_most_accurate_weights(best_performer.get_final_weight())


plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
x_values = [i for i in range(len(descent_result_lipchitz.get_running_distance_to_most_accurate_weights_average()))]
plt.plot(x_values, descent_result_lipchitz.get_running_distance_to_most_accurate_weights_average(), label="a = 1/L")
#plt.plot(x_values, descent_result_0001.get_distances_to_best_weight(), label="a = 0.001")
#plt.plot(x_values, descent_result_001.get_distances_to_best_weight(), label="a = 0.01")
#plt.plot(x_values, descent_result_005.get_distances_to_best_weight(), label="a = 0.05")
#plt.plot(x_values, descent_result_01.get_distances_to_best_weight(), label="a = 0.1")
#plt.plot(x_values, descent_result_05.get_distances_to_best_weight(), label="a = 0.5")
plt.plot(x_values, descent_result_l40.get_running_distance_to_most_accurate_weights_average(), label="a = 1/4")
plt.plot(x_values, descent_result_l50.get_running_distance_to_most_accurate_weights_average(), label="a = 1/5")
plt.plot(x_values, descent_result_l70.get_running_distance_to_most_accurate_weights_average(), label="a = 1/7")
plt.plot(x_values, descent_result_l80.get_running_distance_to_most_accurate_weights_average(), label="a = 1/8")
#plt.plot(x_values, descent_result_07.get_distances_to_best_weight(), label="a = 0.7")
#plt.plot(x_values, descent_result_1.get_distances_to_best_weight(), label="a = 1")
plt.legend(loc='upper right')
plt.show()


dump_array_to_csv(descent_result_lipchitz.get_distances_to_most_accurate_weight(), "descent_result_lipchitz.get_distances_to_best_weight.csv", True, "100k")
dump_array_to_csv(descent_result_lipchitz.get_distances_to_final_weight(), "descent_result_lipchitz.get_distances_to_final_weight.csv", True, "100k")
dump_array_to_csv(descent_result_lipchitz.get_running_distance_to_most_accurate_weights_average(), "descent_result_lipchitz.get_running_distance_to_best_weights_average.csv", True, "100k")
dump_array_to_csv(descent_result_lipchitz.get_best_weight_over_time_distances_to_best_weight(), "descent_result_lipchitz.get_best_weight_over_time_distances_to_best_weight.csv", True, "100k")
dump_array_to_csv(descent_result_lipchitz.get_accuracy_over_time(), "descent_result_lipchitz.get_accuracy_over_time.csv", True, "100k")
dump_array_to_csv(descent_result_lipchitz.get_best_weights_over_time(), "descent_result_lipchitz.get_best_weights_over_time.csv", True, "100k")
dump_array_to_csv(descent_result_lipchitz.get_weights_over_time(), "descent_result_lipchitz.get_best_weights_over_time.csv", True, "100k")
dump_array_to_csv(descent_result_lipchitz.get_most_accurate_weights(), "descent_result_lipchitz.get_best_weights.csv", True)
#
#dump_array_to_csv(descent_result_01.get_distances_to_best_weight(), "descent_result_01.get_distances_to_best_weight.csv", True, "100k")
#dump_array_to_csv(descent_result_01.get_distances_to_final_weight(), "descent_result_01.get_distances_to_final_weight.csv", True, "100k")
#dump_array_to_csv(descent_result_01.get_running_distance_to_best_weights_average(), "descent_result_01.get_running_distance_to_best_weights_average.csv", True, "100k")
#dump_array_to_csv(descent_result_01.get_best_weight_over_time_distances_to_best_weight(), "descent_result_01.get_best_weight_over_time_distances_to_best_weight.csv", True, "100k")
#dump_array_to_csv(descent_result_01.get_accuracy_over_time(), "descent_result_01.get_accuracy_over_time.csv", True, "100k")
#dump_array_to_csv(descent_result_01.get_best_weights_over_time(), "descent_result_01.get_best_weights_over_time.csv", True, "100k")
#dump_array_to_csv(descent_result_01.get_weights_over_time(), "descent_result_01.get_best_weights_over_time.csv", True, "100k")
#dump_array_to_csv(descent_result_01.get_best_weights(), "descent_result_01.get_best_weights.csv", True, "100k")
#
#dump_array_to_csv(descent_result_005.get_distances_to_best_weight(), "descent_result_005.get_distances_to_best_weight.csv", True, "100k")
#dump_array_to_csv(descent_result_005.get_distances_to_final_weight(), "descent_result_005.get_distances_to_final_weight.csv", True, "100k")
#dump_array_to_csv(descent_result_005.get_running_distance_to_best_weights_average(), "descent_result_005.get_running_distance_to_best_weights_average.csv", True, "100k")
#dump_array_to_csv(descent_result_005.get_best_weight_over_time_distances_to_best_weight(), "descent_result_005.get_best_weight_over_time_distances_to_best_weight.csv", True, "100k")
#dump_array_to_csv(descent_result_005.get_accuracy_over_time(), "descent_result_005.get_accuracy_over_time.csv", True, "100k")
#dump_array_to_csv(descent_result_005.get_best_weights_over_time(), "descent_result_005.get_best_weights_over_time.csv", True, "100k")
#dump_array_to_csv(descent_result_005.get_weights_over_time(), "descent_result_005.get_best_weights_over_time.csv", True, "100k")
#dump_array_to_csv(descent_result_005.get_best_weights(), "descent_result_005.get_best_weights.csv", True, "100k")
#
#dump_array_to_csv(descent_result_001.get_distances_to_best_weight(), "descent_result_001.get_distances_to_best_weight.csv", True, "100k")
#dump_array_to_csv(descent_result_001.get_distances_to_final_weight(), "descent_result_001.get_distances_to_final_weight.csv", True, "100k")
#dump_array_to_csv(descent_result_001.get_running_distance_to_best_weights_average(), "descent_result_001.get_running_distance_to_best_weights_average.csv", True, "100k")
#dump_array_to_csv(descent_result_001.get_best_weight_over_time_distances_to_best_weight(), "descent_result_001.get_best_weight_over_time_distances_to_best_weight.csv", True, "100k")
#dump_array_to_csv(descent_result_001.get_accuracy_over_time(), "descent_result_001.get_accuracy_over_time.csv", True, "100k")
#dump_array_to_csv(descent_result_001.get_best_weights_over_time(), "descent_result_001.get_best_weights_over_time.csv", True, "100k")
#dump_array_to_csv(descent_result_001.get_weights_over_time(), "descent_result_001.get_best_weights_over_time.csv", True, "100k")
#dump_array_to_csv(descent_result_001.get_best_weights(), "descent_result_001.get_best_weights.csv", True, "100k")
