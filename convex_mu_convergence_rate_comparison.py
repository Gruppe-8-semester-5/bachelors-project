from matplotlib import pyplot as plt
import numpy as np
from algorithms.adam import Adam
from algorithms.heavy_ball import Heavy_ball
from algorithms.standard_GD import Standard_GD
from algorithms.accelerated_GD_adaptive import Nesterov_acceleration_adaptive
from datasets.winequality.files import wine_X_y
from analysis.lipschitz import lipschitz_binary_neg_log_likelihood_regularized
from test_runner.test_runner_file import Runner
from models.utility import to_torch
import models.logistic_L2 as cur_model

epsilon = 0
iterations = 10000

folder = "figures/"


def save(fname):
    plt.savefig(fname=folder + fname + ".png", format="png")
    plt.close()


X, y = wine_X_y()
n = X.shape[0]

np.random.seed(0)
w0 = cur_model.initial_params(X)
L = lipschitz_binary_neg_log_likelihood_regularized(X, y, cur_model.L2_const)
mu = cur_model.L2_const * 2
step_size = 1 / L
order = ['Standard', 'Heavy ball', 'Adam']
algs = [Standard_GD, Heavy_ball, Adam]

test_set = {
    "w0": w0,
    "GD_params": {"step_size": step_size},
    "alg": algs,
    "model": cur_model,
    "max_iter": iterations,
    "data_set": (X, y),
    "epsilon": epsilon,
}

NAG_recommended = {
    "w0": w0,
    "GD_params": {"step_size": step_size, 'overridden_beta': [None, (np.sqrt(L) - np.sqrt(mu))/(np.sqrt(L) + np.sqrt(mu))]},
    "alg": Nesterov_acceleration_adaptive,
    "model": cur_model,
    "max_iter": iterations,
    "data_set": (X, y),
    "epsilon": epsilon,
}


best_ = {
    "w0": w0,
    "GD_params": {"step_size": step_size},
    "alg": [Nesterov_acceleration_adaptive],
    "model": cur_model,
    "max_iter": 100000,
    "data_set": (X, y),
    "epsilon": epsilon,
}

runner = Runner(dic=test_set)
runner_nag = Runner(dic=NAG_recommended)
results = runner.get_result()
results_nag = runner_nag.get_result()
runner_ = Runner(dic=best_)


def get_min_and_w_star(*results):
    best = float("inf")
    best_w = results[0].get_weights_over_time()[0]
    for res in results:
        losses = res.get_losses_over_time()
        weights = res.get_weights_over_time()
        coord = np.argmin(losses)
        w = weights[coord]
        cur = np.min(losses)
        if cur < best:
            best = cur
            best_w = w
    return best, best_w

smallest_loss, w_star = get_min_and_w_star(*runner_.get_result(), *results, *results_nag)

x_values = [i for i in range(0, iterations)]

for i, alg_name in enumerate(order):
    alg = algs[i]
    result = runner.get_result(alg=alg)[0]
    plt.plot(x_values, result.get_losses_over_time() - smallest_loss, label=f"{alg_name}")

for des, res in runner_nag.get_res_and_description():
    text = ""
    if des['GD_params']['overridden_beta'] is None:
        text = "NAG variable \u03B2"
    else:
        text = "NAG constant \u03B2"
    plt.plot(x_values, res.get_losses_over_time() - smallest_loss, label=text)

plt.legend(loc="upper right")
plt.yscale('log')
plt.xscale('log')
save("convergence_mu_convex_comparison")


lim = 300
for des, res in runner_nag.get_res_and_description():
    text = ""
    if des['GD_params']['overridden_beta'] is None:
        text = "NAG variable \u03B2"
    else:
        text = "NAG constant \u03B2"
    plt.plot(x_values[:lim], (res.get_losses_over_time() - smallest_loss)[:lim], label=text)

plt.plot(x_values[:lim], (results[1].get_losses_over_time() - smallest_loss)[:lim], label="Heavy ball")
factor = cur_model.loss(*to_torch(X, y, w0)) - smallest_loss
y_vals_max = [(factor) * (1 - np.sqrt(mu / L)) ** (k) for k in x_values]
plt.plot(x_values[:lim], y_vals_max[:lim], label=r'$(f(w_0)-f(w^*))\left(1-\sqrt{\frac{\mu}{L}}\right)^k$')
plt.legend(loc="lower left")
plt.yscale('log')
plt.xscale('log')
save("convergence_mu_convex_theoretical_Nag")
