import matplotlib.pyplot as plt
import numpy as np
from datasets.winequality.files import read_wine_data
from analysis.lipschitz import lipschitz_binary_neg_log_likelihood
from datasets.winequality.wine import Wine
from models.logistic_regression import predict
from models.logistic_regression import gradient
from algorithms import simple_gradient_descent
from algorithms import GradientDescentResult
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
print("feature example", example_wine.get_feature_vector())
print("Label example", color_to_label(example_wine.get_color())) 

color_label_list = list(map(lambda wine: color_to_label(wine.get_color()), wines))
color_label_array = np.array(color_label_list)

feature_size = example_wine.get_feature_vector().size

def make_predictions(weights):
    predictions = []
    for wine in wines:
        predictions.append(predict(weights, wine.get_feature_vector()))

    return predictions

start_gradient = np.random.rand(feature_size)
iterations = 100



#Todo: set the best point to one of the tests, so the graph content matches

descent_result_lipchitz: GradientDescentResult = simple_gradient_descent.find_minima(
    start_gradient, 
    1/lipschitz, 
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

descent_result_0001: GradientDescentResult = simple_gradient_descent.find_minima(
    start_gradient, 
    0.001, 
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

descent_result_001: GradientDescentResult = simple_gradient_descent.find_minima(
    start_gradient, 
    0.01, 
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

descent_result_005: GradientDescentResult = simple_gradient_descent.find_minima(
    start_gradient, 
    0.05, 
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

descent_result_01: GradientDescentResult = simple_gradient_descent.find_minima(
    start_gradient, 
    0.1, 
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

descent_result_05: GradientDescentResult = simple_gradient_descent.find_minima(
    start_gradient, 
    0.5, 
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

descent_result_1: GradientDescentResult = simple_gradient_descent.find_minima(
    start_gradient, 
    1, 
    lambda w: gradient(feature_array, color_label_array, w), 
    max_iter=iterations,
    epsilon=1.0e-2,
    accuracy=(lambda w: accuracy(color_label_array, make_predictions(w))))

w = descent_result_lipchitz.get_final_weight()
dataset = read_wine_data()
wines: list[Wine] = list(map(lambda d: Wine(d), dataset))
example_wine = wines[len(wines) - 1].get_feature_vector()
print(wines[len(wines)-1].get_quality())




print(f"Accuracy:{np.round(accuracy(color_label_array, make_predictions(w))* 100,decimals=2)}%")



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
number_of_points = descent_result_lipchitz.number_of_points()

best_lipschitz_point = descent_result_lipchitz.get_best_weights()

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

# Plot moving averages over time
number_of_points = descent_result_lipchitz.number_of_points()
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
x_values = [i for i in range(len(descent_result_lipchitz.get_running_accuracy_average()))]
plt.plot(x_values, descent_result_lipchitz.get_running_accuracy_average(), label="a = 1/L")
plt.plot(x_values, descent_result_0001.get_running_accuracy_average(), label="a = 0.001")
plt.plot(x_values, descent_result_001.get_running_accuracy_average(), label="a = 0.01")
plt.plot(x_values, descent_result_005.get_running_accuracy_average(), label="a = 0.05")
plt.plot(x_values, descent_result_01.get_running_accuracy_average(), label="a = 0.1")
plt.plot(x_values, descent_result_05.get_running_accuracy_average(), label="a = 0.5")
plt.plot(x_values, descent_result_1.get_running_accuracy_average(), label="a = 1")
plt.show()