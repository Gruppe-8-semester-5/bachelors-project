from matplotlib import pyplot as plt
import numpy as np
from algorithms.standard_GD import Standard_GD
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
iterations = 150

# X, y = mnist_train_X_y()

X, y = wine_X_y()
n = X.shape[0]

np.random.seed(0)
w0 = logistic.initial_params(X)
L = lipschitz_binary_neg_log_likelihood(X, y)
used = 1 / L
constants_normal = [3.8, 2, 1, 0.5]
constants_large = [5, 4, 3, 2]
step_sizes_normal = [used * x for x in constants_normal]
step_sizes_large = [used * x for x in constants_large]
test_set = {
    "w0": w0,
    "GD_params": {"step_size": step_sizes_large + step_sizes_normal},
    "alg": [Standard_GD],
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
    "max_iter": 100000,
    "data_set": (X, y),
    "epsilon": epsilon,
    "batch": None,
}

runner = Runner(dic=test_set)
results_normal = runner.get_result(GD_params={"step_size": step_sizes_normal})
results_large = runner.get_result(GD_params={"step_size": step_sizes_large})

runner_ = Runner(dic=best_)
w_star = runner_.get_result()[0].get_weights_over_time()[-1]
smallest_loss = logistic.negative_log_likelihood(*to_torch(X, y, w_star))

x_values = [i for i in range(1, iterations + 1)]

# first = logistic.negative_log_likelihood(*to_torch(X, y, w0)) - smallest_loss
# best = logistic.negative_log_likelihood(*to_torch(X, y, results_normal[2].get_weights_over_time()[-1])) - smallest_loss
# print(L * np.sum((w0-w_star) ** 2)  / 2)
# # L * np.sum((w0-w_star) ** 2)  / 2 * 1 / K = 0.2203
# # K = L * np.sum((w0-w_star) ** 2)  / 2 / 0.2203 =
# print((L * np.sum((w0-w_star) ** 2)  / 2) / best)
# exit()

loss_diff_normal = []
for res in results_normal:
    loss_diff_normal.append(
        [
            logistic.negative_log_likelihood(*to_torch(X, y, x)) - smallest_loss
            for x in res.get_weights_over_time()
        ]
    )

loss_diff_large = []
for res in results_large:
    loss_diff_large.append(
        [
            logistic.negative_log_likelihood(*to_torch(X, y, x)) - smallest_loss
            for x in res.get_weights_over_time()[:100]
        ]
    )

for i, loss in enumerate(loss_diff_normal):
    plt.plot(x_values, loss, label=f"{constants_normal[i]}/L")
plt.legend(loc="center right")
save("convergence_convex_normal")
# plt.show()

for i, loss in enumerate(loss_diff_large):
    plt.plot(x_values[:100], loss_diff_large[i], label=f"{constants_large[i]}/L")
plt.legend(loc="center right")
save("convergence_convex_large")
