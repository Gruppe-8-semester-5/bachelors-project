from matplotlib import pyplot as plt
import numpy as np
from algorithms.standard_GD import Standard_GD
from algorithms.adam import Adam
from datasets.winequality.files import wine_X_y
from analysis.lipschitz import lipschitz_binary_neg_log_likelihood
from test_runner.test_runner_file import Runner
from models.utility import to_torch
import models.logistic_torch as cur_model

epsilon=1.0e-10
iterations = 200

# X, y = mnist_train_X_y()

X, y = wine_X_y()
n = X.shape[0]

np.random.seed(0)
w0 = cur_model.initial_params(X)
L = lipschitz_binary_neg_log_likelihood(X, y)
used = 1 / L
constants_normal = [3.8, 2, 1, 0.5]
constants_large = [2, 3, 4, 5]
step_sizes_normal = [used * x for x in constants_normal]
step_sizes_large = [used * x for x in constants_large]
test_set = {
    'w0': w0,
    'GD_params': {'step_size': step_sizes_large + step_sizes_normal},
    'alg': [Standard_GD],
    'model': cur_model,
    'max_iter': iterations,
    'data_set': (X, y),
    'epsilon':epsilon,
    'batch': None
}

best_ = {
    'w0': w0,
    'GD_params': {'step_size': 1 / L},
    'alg': [Adam],
    'model': cur_model,
    'max_iter': 10000,
    'data_set': (X, y),
    'epsilon':epsilon,
    'batch': None
}

runner = Runner(dic = test_set)
results_normal = runner.get_result(GD_params={'step_size':step_sizes_normal})
results_large = runner.get_result(GD_params={'step_size':step_sizes_large})

runner_ = Runner(dic = best_)
w_star = runner_.get_result()[0].get_weights_over_time()[-1]
smallest_loss = cur_model.negative_log_likelihood(*to_torch(X, y, w_star))

x_values = [i for i in range(1, iterations+1)]

loss_diff_normal = []
for res in results_normal:
    loss_diff_normal.append([cur_model.negative_log_likelihood(*to_torch(X, y, x)) - smallest_loss for x in res.get_weights_over_time()])

loss_diff_large = []
for res in results_large:
    loss_diff_large.append([cur_model.negative_log_likelihood(*to_torch(X, y, x)) - smallest_loss for x in res.get_weights_over_time()[:100]])
# diff = np.sum((w0 - w_star) ** 2)
# y_vals = [L * diff / 2 * 1 / (k) for k in x_values]

# import pyperclip
# pyperclip.copy('\n'.join(map(lambda x: str(x.item()), loss_diff_normal[1])))
for i, loss in enumerate(loss_diff_normal):
    plt.plot(x_values, loss, label=f"{constants_normal[i]}/L")
# plt.plot(x_values, y_vals, label=f"(Worst case)")
plt.legend(loc='center right')
plt.show()

for i, loss in enumerate(loss_diff_large):
    plt.plot(x_values[:100], loss_diff_large[i], label=f"{constants_large[i]}/L")
plt.legend(loc='center right')
plt.show()

plt.plot(x_values, loss_diff_normal[2], label=f"1/L")
# Values found using LoggerPro
y_vals_1_x = [1.539 / (k) for k in x_values]
y_vals_1_sqrt_x = [1.387 / (np.sqrt(k)) for k in x_values]
plt.plot(x_values, y_vals_1_x, label=f"1.539/k")
plt.plot(x_values, y_vals_1_sqrt_x, label=f"1.387/sqrt(k)")
plt.legend(loc='center right')
plt.yscale('log')
plt.show()
