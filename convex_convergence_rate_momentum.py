from matplotlib import pyplot as plt
import numpy as np
from algorithms.heavy_ball import Heavy_ball
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
w0_list = [w0] + [logistic.initial_params(X) for _ in range(100-1)]
L = lipschitz_binary_neg_log_likelihood(X, y)
used = 1 / L
constants_normal = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]
betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
constants_normal = [0.2, 2]
betas = [0.1, 0.9]
step_sizes_normal = [used * x for x in constants_normal]
test_set = {
    "w0": w0_list,
    "GD_params": {"step_size": step_sizes_normal, 'beta': betas},
    "alg": [Heavy_ball],
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
# results_desc = runner.get_res_and_description()
# results_normal = runner.get_result()

runner_ = Runner(dic=best_)
w_star = runner_.get_result()[0].get_weights_over_time()[-1]
smallest_loss = logistic.negative_log_likelihood(*to_torch(X, y, w_star))

x_values = [i for i in range(1, iterations + 1)]

# loss_diff_normal = []
# for res in results_normal:
#     loss_diff_normal.append(
#         res.get_losses_over_time()
#     )

plot_this = np.zeros(shape=(len(constants_normal), len(betas)))
for i in range(len(constants_normal)):
    for j in range(len(betas)):
        a = step_sizes_normal[i]
        b = betas[j]
        results = runner.get_result(GD_params={"step_size": a, 'beta': b})
        n = len(results)
        mean = 0
        for k in range(n):
            mean += np.mean(results[k].get_losses_over_time())
        mean /= n
        plot_this[i][j] = mean
fig, ax = plt.subplots()
im = ax.imshow(plot_this)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(betas)), labels=betas)
ax.set_yticks(np.arange(len(step_sizes_normal)), labels=step_sizes_normal)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(step_sizes_normal)):
    for j in range(len(betas)):
        text = ax.text(j, i, plot_this[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Harvest of local farmers (in tons/year)")
fig.tight_layout()
plt.show()
exit()
# for i, (desc, res) in enumerate(results_desc):
#     a = desc['GD_params']['step_size']
#     b = desc['GD_params']['beta']
#     plt.plot(x_values, loss_diff_normal[i], label=f"a={L*a}/L b={b} ")
# plt.plot(x_values, y_vals, label=f"(Worst case)")
plt.legend(loc="center right")
# plt.savefig(fname='Test.png', format='png')
# save("convergence_convex_normal")
plt.show()

# plt.plot(x_values, loss_diff_normal[2], label="1/L")
# # Values found using LoggerPro
# y_vals_1_x = [1.694 / (k) for k in x_values]
# y_vals_1_sqrt_x = [1.623 / (np.sqrt(k)) for k in x_values]
# plt.plot(x_values, y_vals_1_x, label="1.694/k")
# plt.plot(x_values, y_vals_1_sqrt_x, label="1.623/sqrt(k)")
# plt.legend(loc="center right")
# plt.yscale("log")
# save("convergence_convex_best_fit")
# plt.savefig(fname=folder + 'convergence_convex_best_fit.png', format='png')
# plt.show()
