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
iterations = 150

X, y = wine_X_y()
n = X.shape[0]

np.random.seed(0)
w0 = logistic.initial_params(X)
w0_list = [w0] + [logistic.initial_params(X) for _ in range(100-1)]
L = lipschitz_binary_neg_log_likelihood(X, y)
used = 1 / L
beta_range_small = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
beta_range_large = [0.999, 0.99, 0.95, 0.9, 0.85, 0.8, 0.7]

step_size = used
test_small = {
    "w0": w0_list,
    "GD_params": {"step_size": step_size, 'b1': beta_range_small, 'b2': beta_range_small},
    "alg": [Adam],
    "model": logistic,
    "max_iter": iterations,
    "data_set": (X, y),
    "epsilon": epsilon,
}

test_large = {
    "w0": w0_list,
    "GD_params": {"step_size": step_size, 'b1': beta_range_large, 'b2': beta_range_large},
    "alg": [Adam],
    "model": logistic,
    "max_iter": iterations,
    "data_set": (X, y),
    "epsilon": epsilon,
}

runner_small = Runner(dic=test_small)
results_small = runner_small.get_result()


runner_large = Runner(dic=test_large)
results_large = runner_large.get_result()

# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
plot_this = np.zeros(shape=(len(beta_range_small), len(beta_range_small)))
for j in range(len(beta_range_small)):
    for l in range(len(beta_range_small)):
        # step_size = step_sizes_normal[i]
        b1 = beta_range_small[j]
        b2 = np.flip(beta_range_small)[l]
        results = runner_small.get_result(GD_params={'step_size': step_size, 'b1': b1, 'b2': b2})
        n = len(results)
        mean = 0
        for k in range(n):
            mean += results[k].get_losses_over_time()[-1]
        mean /= n
        plot_this[j][l] = mean
fig, ax = plt.subplots()
im = ax.imshow(plot_this)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(beta_range_small)), labels=np.flip(beta_range_small))
ax.set_yticks(np.arange(len(beta_range_small)), labels=beta_range_small)
plt.xlabel("\u03B2_2")
plt.ylabel("\u03B2_1")

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

ax.set_title("Adam for small values of \u03B2_1 and \u03B2_2")
fig.tight_layout()
cbar = ax.figure.colorbar(im, ax=ax)
save('Adam_heat_mean_small')

plot_this = np.zeros(shape=(len(beta_range_large), len(beta_range_large)))
for j in range(len(beta_range_large)):
    for l in range(len(beta_range_large)):
        b1 = beta_range_large[j]
        b2 = np.flip(beta_range_large)[l]
        results = runner_large.get_result(GD_params={'step_size': step_size, 'b1': b1, 'b2': b2})
        n = len(results)
        mean = 0
        for k in range(n):
            mean += results[k].get_losses_over_time()[-1]
        mean /= n
        plot_this[j][l] = mean
fig, ax = plt.subplots()
im = ax.imshow(plot_this)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(beta_range_large)), labels=np.flip(beta_range_large))
ax.set_yticks(np.arange(len(beta_range_large)), labels=beta_range_large)
plt.xlabel("\u03B2_2")
plt.ylabel("\u03B2_1")

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

ax.set_title("Adam for large values of \u03B2_1 and \u03B2_2")
fig.tight_layout()
cbar = ax.figure.colorbar(im, ax=ax)
save('Adam_heat_mean_large')
