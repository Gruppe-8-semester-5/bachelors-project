from matplotlib import pyplot as plt
import numpy as np
from algorithms.accelerated_GD_adaptive import Nesterov_acceleration_adaptive
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

constants_normal = [0.2, 2.]
alphas = [0.9, 0.1]
step_sizes_normal = [used * x for x in constants_normal]
test_set = {
    "w0": w0,
    "GD_params": {"step_size": step_sizes_normal, 'start_alpha': alphas},
    "alg": [Nesterov_acceleration_adaptive],
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

for c in constants_normal:
    for alpha in alphas:
        step_size = c / L
        result = runner.get_result(GD_params={"step_size": step_size, 'start_alpha': alpha})[0]
        plt.plot(x_values, result.get_losses_over_time() - smallest_loss.item(), label=f"\u03B1={c}/L t_0={alpha}")
plt.legend(loc="center right")
plt.yscale('log')
plt.ylim(0.06, 1)
# plt.show()
save("convergence_convex_nesterov_typical_runs")
