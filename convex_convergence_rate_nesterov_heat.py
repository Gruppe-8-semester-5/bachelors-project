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
iterations = 150

# X, y = mnist_train_X_y()

X, y = wine_X_y()
n = X.shape[0]

np.random.seed(0)
w0 = logistic.initial_params(X)
w0_list = [w0] + [logistic.initial_params(X) for _ in range(100-1)]
L = lipschitz_binary_neg_log_likelihood(X, y)
used = 1 / L
constants_normal = [2, 1.8, 1.6, 1.4, 1.2, 1., 0.8, 0.6, 0.4, 0.2]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
step_sizes_normal = [used * x for x in constants_normal]
test_set = {
    "w0": w0_list,
    "GD_params": {"step_size": step_sizes_normal, 'start_alpha': alphas},
    "alg": [Nesterov_acceleration_adaptive],
    "model": logistic,
    "max_iter": iterations,
    "data_set": (X, y),
    "epsilon": epsilon,
    "batch": None,
}

runner = Runner(dic=test_set)
# results_desc = runner.get_res_and_description()
results_normal = runner.get_result()



# We get an accuracy, at early stop (using L2 with L2_const=50 and standard parameters without 1/L2_const multiplied):
# 0.8591657688163767

# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
plot_this = np.zeros(shape=(len(constants_normal), len(alphas)))
for i in range(len(constants_normal)):
    for j in range(len(alphas)):
        step_size = step_sizes_normal[i]
        a = alphas[j]
        results = runner.get_result(GD_params={"start_alpha": a, 'step_size': step_size})
        n = len(results)
        mean = 0
        for k in range(n):
            mean += np.mean(results[k].get_losses_over_time())
        mean /= n
        plot_this[i][j] = mean
fig, ax = plt.subplots()
im = ax.imshow(plot_this)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(alphas)), labels=alphas)
ax.set_yticks(np.arange(len(constants_normal)), labels=np.char.add(np.array(constants_normal).astype('str'), "/L"))
plt.xlabel("t_0")
plt.ylabel("\u03B1")

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
# for i in range(len(step_sizes_normal)):
#     for j in range(len(betas)):
#         text = ax.text(j, i, plot_this[i, j],
#                        ha="center", va="center", color="w")

ax.set_title("Mean loss alpha/step size")
fig.tight_layout()
cbar = ax.figure.colorbar(im, ax=ax)
# cbar.ax.set_ylabel('some text', rotation=-90, va="bottom")
# plt.show()
# plt.show()
save('nesterov_heat_mean_all')


# plot_this = np.zeros(shape=(len(constants_normal), len(alphas)))
# for i in range(len(constants_normal)):
#     for j in range(len(alphas)):
#         step_size = step_sizes_normal[i]
#         a = alphas[j]
#         results = runner.get_result(GD_params={"start_alpha": a, 'step_size': step_size})
#         n = len(results)
#         mean = 0
#         for k in range(n):
#             mean += np.mean(results[k].get_losses_over_time()[:10])
#         mean /= n
#         plot_this[i][j] = mean
# fig, ax = plt.subplots()
# im = ax.imshow(plot_this)

# # Show all ticks and label them with the respective list entries
# ax.set_xticks(np.arange(len(alphas)), labels=alphas)
# ax.set_yticks(np.arange(len(constants_normal)), labels=np.char.add(np.array(constants_normal).astype('str'), "/L"))

# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")

# # Loop over data dimensions and create text annotations.
# # for i in range(len(step_sizes_normal)):
# #     for j in range(len(betas)):
# #         text = ax.text(j, i, plot_this[i, j],
# #                        ha="center", va="center", color="w")

# ax.set_title("Mean over first 10 losses")
# fig.tight_layout()
# cbar = ax.figure.colorbar(im, ax=ax)
# # cbar.ax.set_ylabel('some text', rotation=-90, va="bottom")
# # save('momentum_heat_first_10')
# plt.show()


plot_this = np.zeros(shape=(len(constants_normal), len(alphas)))
for i in range(len(constants_normal)):
    for j in range(len(alphas)):
        step_size = step_sizes_normal[i]
        alpha = alphas[j]
        results = runner.get_result(GD_params={"start_alpha": alpha, 'step_size': step_size})
        n = len(results)
        mean = 0
        for k in range(n):
            mean += results[k].get_losses_over_time()[-1]
        mean /= n
        plot_this[i][j] = mean
fig, ax = plt.subplots()
im = ax.imshow(plot_this)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(alphas)), labels=alphas)
ax.set_yticks(np.arange(len(constants_normal)), labels=np.char.add(np.array(constants_normal).astype('str'), "/L"))
plt.xlabel("t_0")
plt.ylabel("\u03B1")

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
# for i in range(len(step_sizes_normal)):
#     for j in range(len(betas)):
#         text = ax.text(j, i, plot_this[i, j],
#                        ha="center", va="center", color="w")

ax.set_title("Last element")
fig.tight_layout()
cbar = ax.figure.colorbar(im, ax=ax)
# cbar.ax.set_ylabel('some text', rotation=-90, va="bottom")
# plt.show()
save('nesterov_heat_last')
