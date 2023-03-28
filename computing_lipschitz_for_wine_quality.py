import numpy as np
from datasets.winequality.files import read_wine_data
from analysis.lipschitz import lipschitz_binary_neg_log_likelihood
from datasets.winequality.wine import Wine

# Construct np array of features
dataset = read_wine_data()
wines: list[Wine] = list(map(lambda d: Wine(d), dataset))
feature_list = list(map(lambda wine: wine.get_feature_vector(), wines))
feature_array = np.array(feature_list)

print("Lipschitz - constant for negative log-likelihood")
print("L=(1/2n)\sum_{i=1}^n ||{x_i}||^2")
n = len(dataset)
print("For our wine quality dataset")
print("n = ", n)

feature_norm_sum = 0
for features in feature_array:
    feature_norm_sum += np.linalg.norm(features)

print("sum ||x_i||^2 = ", feature_norm_sum)

print("Giving us L = ", lipschitz_binary_neg_log_likelihood(feature_array))