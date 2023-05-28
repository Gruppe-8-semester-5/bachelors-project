from matplotlib import pyplot as plt
import numpy as np
from algorithms.adam import Adam
from datasets.winequality.files import wine_X_y
from analysis.lipschitz import lipschitz_binary_neg_log_likelihood
from test_runner.test_runner_file import Runner
from models.utility import to_torch
import models.logistic_torch as logistic

folder = "figures/"


def save(fname):
    plt.savefig(fname=folder + fname + ".png", format="png")
    plt.close()

epsilon = 1.0e-10
iterations = 1000

X, y = wine_X_y()
n = X.shape[0]

np.random.seed(0)
w0 = logistic.initial_params(X)
L = lipschitz_binary_neg_log_likelihood(X, y)
used = 1 / L

beta_pairs = [(0.1, 0.1), (0.999, 0.999), (0.7, 0.7), (0.94, 0.9879)]
beta_1 = [b for b, _ in beta_pairs]
# [0.999, 0.99, 0.95, 0.9, 0.85, 0.8, 0.7]
beta_2 = [b for _, b in beta_pairs]
step_size = used
test_set = {
    "w0": w0,
    "GD_params": {"step_size": step_size, 'b1': beta_1, 'b2': beta_2},
    "alg": [Adam],
    "model": logistic,
    "max_iter": iterations,
    "data_set": (X, y),
    "epsilon": epsilon,
    "batch": None,
}

best_ = {
    "w0": w0,
    "GD_params": {"step_size": 1 / L},
    "alg": [Adam],
    "model": logistic,
    "max_iter": 10000,
    "data_set": (X, y),
    "epsilon": epsilon,
    "batch": None,
}

runner = Runner(dic=test_set)
runner_ = Runner(dic=best_)
w_star = runner_.get_result()[0].get_weights_over_time()[-1]
smallest_loss = logistic.negative_log_likelihood(*to_torch(X, y, w_star))

x_values = [i for i in range(1, iterations + 1)]
results = runner.get_result()

for b1, b2 in beta_pairs:
    result = runner.get_result(GD_params={"step_size": step_size, 'b1': b1, 'b2': b2})[0]
    plt.plot(x_values, result.get_losses_over_time() - smallest_loss.item(), label=f"\u03B2_1={b1} \u03B2_2={b2}")
plt.legend(loc="center right")
plt.yscale('log')
plt.ylim(0.06, 1)
save("convergence_convex_adam_typical_runs")
