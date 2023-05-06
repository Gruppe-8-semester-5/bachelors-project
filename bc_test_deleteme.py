from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from algorithms import gradient_descent_template
from algorithms.gradient_descent_result import GradientDescentResult
from algorithms.standard_GD import Standard_GD
from models.logistic_regression import gradient, negative_log_likelihood, predict
from models.utility import accuracy
data = load_breast_cancer()


features = np.array(data['data'])
labels = data['target']
labels = np.array([-1. if l == 0 else 1. for l in labels])

print(f"features {features}")
print(f"labels {labels}")

from analysis.lipschitz import lipschitz_binary_neg_log_likelihood

shitz = lipschitz_binary_neg_log_likelihood(features, labels)

print(shitz)

def make_predictions(weights):
    predictions = []
    for feature in features:
        predictions.append(predict(weights, feature))
    return predictions

example = features[0]

start_gradient = np.random.normal(example)

descent_result_lipchitz: GradientDescentResult = gradient_descent_template.find_minima(
    start_gradient, 
    Standard_GD(1/shitz),
    lambda w: gradient(features, labels, w), 
    max_iter=500,
    epsilon=1.0e-100,
    accuracy=(lambda w: accuracy(labels, make_predictions(w))))

print(descent_result_lipchitz.get_best_accuracy())

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
x_values = [i for i in range(len(descent_result_lipchitz.get_accuracy_over_time()))]
plt.plot(x_values, descent_result_lipchitz.get_accuracy_over_time(), label="a = 1/L")
plt.legend(loc='upper left')
plt.show()