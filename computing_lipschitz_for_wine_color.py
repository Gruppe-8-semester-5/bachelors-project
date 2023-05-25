import numpy as np
from datasets.winequality.files import read_wine_data
from analysis.lipschitz import lipschitz_binary_neg_log_likelihood
from datasets.winequality.wine import Wine

# Construct np array of features
dataset = read_wine_data()
wines: list[Wine] = list(map(lambda d: Wine(d), dataset))
feature_list = list(map(lambda wine: wine.get_feature_vector(), wines))
feature_array = np.array(feature_list)


# Array of labels
def color_to_label(color):
    if color == "white":
        return 0
    return 1

label_list = list(map(lambda wine: color_to_label(wine.get_color()), wines))
label_array = np.array(label_list)

print("Lipschitz - constant for negative log-likelihood")
print("L=(1/2n)\sum_{i=1}^n ||{x_i}||^2")
n = len(dataset)
print("For our wine quality dataset")
print("n = ", n)

feature_norm_sum = 0
n = len(feature_array)
for i in range(0, n):
    features = feature_array[i]
    feature_norm_sum += np.linalg.norm(features) ** 2

print("sum ||x_i||^2 = ", feature_norm_sum)

print("Giving us L = ", lipschitz_binary_neg_log_likelihood(feature_array, label_array))