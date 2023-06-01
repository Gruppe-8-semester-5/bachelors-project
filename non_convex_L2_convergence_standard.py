from matplotlib import pyplot as plt
import numpy as np
from algorithms.adam import Adam
from algorithms.standard_GD import Standard_GD
from datasets.mnist.files import mnist_train_X_y, mnist_test_X_y
from test_runner.test_runner_file import Runner
import models.one_hidden_relu_softmax_L2 as model

epsilon = 0
iterations = 2
folder = "figures/"


def save(fname):
    plt.savefig(fname=folder + fname + ".png", format="png")
    plt.close()

X_train, y_train = mnist_train_X_y()

X_test, y_test = mnist_test_X_y()
n = X_train.shape[0]

np.random.seed(0)
w0 = model.initial_params(X_train.shape[1], 300, np.max(y_train) + 1)
step_sizes = [0.1, 0.05, 0.01, 0.001, 0.0001, 0.00005, 0.00001]

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

adam_set = {
    "w0": w0,
    "GD_params": {"step_size": 0.01},
    "alg": Adam,
    "model": model,
    "max_iter": 1000,
    "data_set": (X_train, y_train),
    "epsilon": epsilon,
    "batch": None,
}

runner_adam = Runner(dic=adam_set)
result_adam = runner_adam.get_result()[0]
losses = result_adam.get_losses_over_time()

runner = Runner(dic=test_set)
results = runner.get_result()

x_values = [i for i in range(1, iterations+1)]

for step_size in step_sizes:
    result = runner.get_result(GD_params={'step_size': step_size})[0]
    plt.plot(x_values, result.get_losses_over_time(), label=f"\u03B1={step_size}")

plt.legend(loc="lower left")
plt.yscale('log')
plt.xscale('log')
# plt.show()
save("convergence_non_convex_L2_loss_standard")

f_0 = losses[0]
f_best = losses[-1]
alpha=0.1
L=1
const = np.sqrt(-1/(-alpha+alpha**2*L/2)*(f_0-f_best))
bound = const/np.sqrt(x_values)
print('f0', f_0)
print('f_best', f_best)
plt.plot(x_values, bound, label=f"Bound for \u03B1={alpha} and L={L}")

for step_size in step_sizes:
    result = runner.get_result(GD_params={'step_size': step_size})[0]
    plt.plot(x_values, result.get_grad_norms_over_time(), label=f"\u03B1={step_size}")

plt.legend(loc="lower left")
plt.yscale('log')
# plt.xscale('log')
save("convergence_non_convex_L2_norm_standard")
# plt.show()

# Fantastic python-y code.
# import pyperclip
# grad_norms = []
# for step_size in step_sizes:
#     result = runner.get_result(GD_params={'step_size': step_size})[0]
#     grad_norms.append([str(g) for g in result.get_grad_norms_over_time()])

# zipped = list(zip(*grad_norms))
# text = '\n'.join(['\t'.join(grads) for grads in zipped])
# pyperclip.copy(text)
