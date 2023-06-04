from matplotlib import pyplot as plt
import numpy as np
from algorithms.adam import Adam
from algorithms.heavy_ball import Heavy_ball
from algorithms.standard_GD import Standard_GD
from algorithms.accelerated_GD_adaptive import Nesterov_acceleration_adaptive
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
iterations = 10000

X, y = wine_X_y()
n = X.shape[0]

np.random.seed(0)
w0 = logistic.initial_params(X)
L = lipschitz_binary_neg_log_likelihood(X, y)

order = ['Standard', 'Heavy ball', 'NAG', 'Adam']
algs = [Standard_GD, Heavy_ball, Nesterov_acceleration_adaptive, Adam]
step_size = 1 / L
test_set = {
    "w0": w0,
    "GD_params": {"step_size": step_size},
    "alg": algs,
    "model": logistic,
    "max_iter": iterations,
    "data_set": (X, y),
    "epsilon": epsilon,
}

best_ = {
    "w0": w0,
    "GD_params": {"step_size": 1 / L},
    "alg": [Adam],
    "model": logistic,
    "max_iter": 100000,
    "data_set": (X, y),
    "epsilon": epsilon,
}

runner = Runner(dic=test_set)
runner_ = Runner(dic=best_)
smallest_loss = np.min(runner_.get_result()[0].get_losses_over_time())
x_values = [i for i in range(1, iterations + 1)]
results = runner.get_result()

for i, alg_name in enumerate(order):
    alg = algs[i]
    result = runner.get_result(alg=alg)[0]
    plt.plot(x_values, result.get_losses_over_time() - smallest_loss, label=f"{alg_name}")
plt.legend(loc="upper right")
plt.xscale('log')
plt.ylim(0, 0.8)
plt.show()

save("convergence_convex_comparison")
