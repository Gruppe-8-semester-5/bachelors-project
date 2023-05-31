from matplotlib import pyplot as plt
import numpy as np
from algorithms.standard_GD import Standard_GD
from datasets.mnist.files import mnist_train_X_y, mnist_test_X_y
from test_runner.test_runner_file import Runner
import models.one_hidden_relu_softmax_L2 as model

epsilon = 0
iterations = 1000
folder = "figures/"


def save(fname):
    plt.savefig(fname=folder + fname + ".png", format="png")
    plt.close()

X_train, y_train = mnist_train_X_y()

X_test, y_test = mnist_test_X_y()
n = X_train.shape[0]

np.random.seed(0)
w0 = model.initial_params(X_train.shape[1], 300, np.max(y_train) + 1)
step_sizes = [0.01, 0.001, 0.0001, 0.00005, 0.00001]

test_set = {
    "w0": w0,
    "GD_params": {"step_size": step_sizes},
    "alg": Standard_GD,
    "model": model,
    "max_iter": iterations,
    "data_set": (X_train, y_train),
    "epsilon": epsilon,
    "batch": None,
}

runner = Runner(dic=test_set)
results = runner.get_result()

x_values = [i for i in range(0, iterations)]

for step_size in step_sizes:
    result = runner.get_result(GD_params={'step_size': step_size})[0]
    plt.plot(x_values, result.get_losses_over_time(), label=f"\u03B1={step_size}")

plt.legend(loc="lower left")
plt.yscale('log')
# plt.xscale('log')
# plt.show()
save("convergence_non_convex_L2_loss_standard")


for step_size in step_sizes:
    result = runner.get_result(GD_params={'step_size': step_size})[0]
    plt.plot(x_values, result.get_grad_norms_over_time(), label=f"\u03B1={step_size}")

plt.legend(loc="lower left")
plt.yscale('log')
# plt.xscale('log')
save("convergence_non_convex_L2_norm_standard")
# plt.show()
