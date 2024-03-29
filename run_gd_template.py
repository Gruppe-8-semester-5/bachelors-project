from matplotlib import pyplot as plt
import numpy as np
from algorithms import GradientDescentResult, gradient_descent_template, standard_GD
from algorithms.standard_GD import Standard_GD
from algorithms.momentum_GD import Momentum
from algorithms.adam import Adam
from datasets.mnist.files import mnist_train_X_y
from datasets.winequality.files import wine_X_y
from analysis.lipschitz import lipschitz_binary_neg_log_likelihood
from test_runner.test_runner_file import Runner
from models.utility import to_torch

import models.logistic_torch as logistic

epsilon = 1.0e-10
iterations = 1000

X, y = wine_X_y()
n = X.shape[0]

np.random.seed(0)
output_shape = 0
output_shape = np.amax(y) + 1

# w0 = cur_model.initial_params(X.shape[1], output_shape)
# w0 = cur_model.initial_params(X.shape[1], 100, output_shape)
w0 = logistic.initial_params(X)
# w0 = [cur_model.initial_params(X) for _ in range(1)]
# grad = lambda w: gradient(X, y, w)
# List of things we want to test. Form (optimizer, params)
L = lipschitz_binary_neg_log_likelihood(X, y)
used = 1 / L
# test_set = {
#     'w0': w0,
#     'GD_params': {'step_size': used},
#     # 'GD_params': {'L': [0.01], 'w0': w0},
#     'alg': [Standard_GD],
#     'model': cur_model,
#     'max_iter': iterations,
#     'data_set': (X, y),
#     'epsilon':epsilon,
#     'batch': [5, 10, 20, 100, None]
#     # 'batch': [1, 5, 10, 20, None]
# }
test_set = {
    "w0": w0,
    "GD_params": {"step_size": [used * 3.8, used * 2, used, used / 2]},
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
    "max_iter": iterations,
    "data_set": (X, y),
    "epsilon": epsilon,
    "batch": None,
}


runner = Runner(dic=test_set)
results_and_des = runner.get_res_and_description()
results = [x for _, x in results_and_des]
print(results[0].get_best_accuracy())


runner_ = Runner(dic=best_)

smallest_loss = float("inf")
w_star = 0
x_values = [
    i
    for i in range(
        0, len(results[0].get_best_weight_over_time_distances_to_best_weight())
    )
]
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

losses = []
for res in results:
    losses.append(
        [
            logistic.negative_log_likelihood(*to_torch(X, y, x))
            for x in res.get_weights_over_time()
        ]
    )


w_star = runner_.get_result()[0].get_weights_over_time()[-1]
smallest_loss = logistic.negative_log_likelihood(*to_torch(X, y, w_star))

# for i, loss in enumerate(losses):
#     for j, l in enumerate(loss):
#         if l < smallest_loss:
#             smallest_loss = l
#             w_star = results[i].get_weights_over_time()[j]
# if min(loss) < smallest_loss:
#     smallest_loss = min(loss)

diff = np.sum((w0 - w_star) ** 2)
y_vals = [L * diff / 2 * 1 / (k + 1) for k in x_values]

print(L * diff / 2 * 1)
exit()

loss_diff = [[loss - smallest_loss for loss in x] for x in losses]
import pyperclip

pyperclip.copy("\n".join(map(lambda x: str(x.item()), loss_diff[1])))
exit()
for i, (des, res) in enumerate(results_and_des):
    # print(res)
    # result = res['result']
    # is_worst = result == worst_performer
    # result.set_most_accurate_weights(best_performer.get_most_accurate_weights())
    # if is_worst:
    #     plt.plot(x_values, result.get_distances_to_most_accurate_weight(), label=f"Worst performer ({res['mu']},{res['L']}, {res['alpha']}, {res['beta']})", color='r')
    # else:
    # print(loss_diff)
    # exit()
    # plt.plot(x_values, loss_diff[i], label=f"({des['batch']})")
    plt.plot(x_values, loss_diff[i], label=f"({des['GD_params']['step_size']})")
# plt.plot(x_values, y_vals, label=f"(Worst case)")
plt.legend(loc="center right")
# plt.yscale('log')
plt.show()


# exit()
# for r in results:
#     print(r.to_serialized())
exit()

test_set = {
    "w0": w0,
    "GD_params": {"step_size": [0.01, 0.05, 0.1, 0.5, 1]},
    "alg": [Standard_GD, Momentum],
    "model": normal_logistic,
    "max_iter": iterations,
    "data_set": (X, y),
    "epsilon": 1.0e-2,
    "batch": None,
}

runner = Runner(dic=test_set)

results = runner.get_result(alg=Standard_GD)
