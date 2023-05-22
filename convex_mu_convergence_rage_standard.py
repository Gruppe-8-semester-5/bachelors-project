from matplotlib import pyplot as plt
import numpy as np
from algorithms.standard_GD import Standard_GD
from algorithms.adam import Adam
from algorithms.accelerated_GD_adaptive import Nesterov_acceleration_adaptive
from datasets.winequality.files import wine_X_y
from analysis.lipschitz import lipschitz_binary_neg_log_likelihood_regularized
from test_runner.test_runner_file import Runner
from models.utility import to_torch
import models.logistic_L2 as cur_model
import models.logistic_torch as old_model

epsilon=1.0e-15
iterations = 1000

# X, y = mnist_train_X_y()

X, y = wine_X_y()
n = X.shape[0]

np.random.seed(0)
w0 = cur_model.initial_params(X)
L = lipschitz_binary_neg_log_likelihood_regularized(X, y, cur_model.L2_const)
mu = cur_model.L2_const * 2
used = 1 / L
constants_normal = [3.8, 2, 1, 0.5]
step_sizes_normal = [used * x for x in constants_normal]
test_set = {
    'w0': w0,
    'GD_params': {'step_size': step_sizes_normal},
    'alg': [Standard_GD],
    'model': cur_model,
    'max_iter': iterations,
    'data_set': (X, y),
    'epsilon':epsilon,
    'batch': None
}

best_ = {
    'w0': w0,
    'GD_params': {'L': L, 'w0': w0},
    'alg': [Nesterov_acceleration_adaptive],
    'model': cur_model,
    'max_iter': 10000,
    'data_set': (X, y),
    'epsilon':epsilon,
    'batch': None
}

runner = Runner(dic = test_set)
results_normal = runner.get_result(GD_params={'step_size':step_sizes_normal})
runner_ = Runner(dic = best_)
# print(min([cur_model.loss(*to_torch(X,y,w)) for w in runner_.get_result()[0].get_weights_over_time()]).item())
def get_min_and_w_star(*results):
    weights = [[w for w in res.get_weights_over_time()] for res in results]
    best = float('inf')
    best_w = weights[0][0]
    for w_list in weights:
        for w in w_list:
            loss = cur_model.loss(*to_torch(X, y, w))
            if loss < best:
                best_w = w
                best = loss
    return best, best_w
smallest_loss, w_star = get_min_and_w_star(*results_normal, *runner_.get_result())
# weights = [[w for w in res.get_weights_over_time()] for res in results_normal + results_large + runner_.get_result()]
# losses = [[cur_model.loss(*to_torch(X, y, w)).item() for w in res] for res in weights]
# loss_np = np.array(losses)
# smallest_loss = np.min(loss_np)
# print(np.unravel_index(np.argmin(loss_np), shape=loss_np.shape))
# exit()
# i, j = np.unravel_index(np.argmin(loss_np), shape=loss_np.shape)
# w_star = weights[i][j]
# smallest_loss = cur_model.loss(*to_torch(X, y, w_star))

# w_star = runner_.get_result()[0].get_weights_over_time()[-1]

# print(cur_model.loss(*to_torch(X, y, w_star)))
# print(smallest_loss)
# exit()
# print(results_normal[0].get_best_accuracy())
print(runner_.get_result()[0].get_best_accuracy())

# We can get an accuracy of (using L2 with L2_const=50 and standard parameters without 1/L2_const multiplied):
# 0.9124211174388179
# 0.9255040788056026

x_values = [i for i in range(0, iterations)]

loss_diff_normal = []
for res in results_normal:
    loss_diff_normal.append([cur_model.loss(*to_torch(X, y, x)) - smallest_loss for x in res.get_weights_over_time()])

# diff = np.sum((w0 - w_star) ** 2)
# y_vals = [L * diff / 2 * 1 / (k) for k in x_values]

# print(cur_model.loss(*to_torch(X, y, w0)) - cur_model.loss(*to_torch(X, y, w_star)))

# print(mu)
# print(L)
# print(smallest_loss)
# exit()
# import pyperclip
# pyperclip.copy('\n'.join(map(lambda x: str(x.item()), loss_diff_normal[2])))
# exit()
for i, loss in enumerate(loss_diff_normal):
    # from models.utility import accuracy
    # print(loss)
    # print(min([cur_model.loss(*to_torch(X, y, x)) - smallest_loss for x in results_normal[0].get_weights_over_time()]))
    # print(accuracy(y, cur_model.predict(results_normal[0].get_best_weights_over_time()[-1], X)))
    plt.plot(x_values, loss, label=f"{constants_normal[i]}/L")
# plt.plot(x_values, y_vals, label=f"(Worst case)")
plt.legend(loc='center right')
plt.show()
exit()

plt.plot(x_values, loss_diff_normal[2], label=f"1/L")
# Values found using LoggerPro
ratio = 1-mu/L
factor = cur_model.loss(*to_torch(X, y, w0)) - smallest_loss
y_vals_max = [(factor) * (1-mu/L)**(k) for k in x_values]
# y_vals_1_sqrt_x = [1.387 / (np.sqrt(k)) for k in x_values]

# print(y_vals_max[0])
# print(loss_diff_normal[2][0])
# exit()
# print(torch.argwhere(torch.tensor(y_vals_max) <= torch.tensor(loss_diff_normal[2])))
plt.plot(x_values, y_vals_max, label=f"Theoretical limit")
# plt.plot(x_values, y_vals_1_sqrt_x, label=f"1.387/sqrt(k)")
plt.legend(loc='center right')
# plt.yscale('log')
plt.show()
