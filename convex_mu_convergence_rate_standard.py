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

epsilon=1.0e-11
iterations = 10000

folder = 'figures/'
def save(fname):
    plt.savefig(fname=folder + fname + '.png', format='png')
    plt.close()

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

print(runner_.get_result()[0].get_accuracy_over_time()[-1])

# We get an accuracy, at early stop (using L2 with L2_const=50 and standard parameters without 1/L2_const multiplied):
# 0.8591657688163767

x_values = [i for i in range(0, iterations)]

loss_diff_normal = []
for res in results_normal:
    loss_diff_normal.append([cur_model.loss(*to_torch(X, y, x)) - smallest_loss for x in res.get_weights_over_time()])

# print(mu)
# print(L)
# exit()
for i, loss in enumerate(loss_diff_normal):
    plt.plot(x_values[:400], loss[:400], label=f"{constants_normal[i]}/L")
plt.legend(loc='center right')
save('convergence_mu_convex_normal')

for i, loss in enumerate(loss_diff_normal):
    plt.plot(x_values[600:700], loss[600:700], label=f"{constants_normal[i]}/L")

plt.legend(loc='center right')
save('convergence_mu_convex_zoomed')
lim = 400
plt.plot(x_values[:lim], loss_diff_normal[2][:lim], label=f"1/L")
ratio = 1-mu/L
factor = cur_model.loss(*to_torch(X, y, w0)) - smallest_loss
y_vals_max = [(factor) * (1-mu/L)**(k) for k in x_values]

# print(torch.argwhere(torch.tensor(y_vals_max) <= torch.tensor(loss_diff_normal[2])))
plt.plot(x_values[:lim], y_vals_max[:lim], label=f"Theoretical limit")
plt.legend(loc='center right')
save('convergence_mu_convex_theoretical')

# print(results_normal[0].get_weights_over_time()[600] - results_normal[0].get_weights_over_time()[602])

# print(results_normal[1].get_accuracy_over_time()[-1])
# print(results_normal[2].get_accuracy_over_time()[-1])
# print(results_normal[1].get_best_accuracy())
# print(results_normal[2].get_best_accuracy())
print(results_normal[1].get_weights_over_time()[-1])

