from matplotlib import pyplot as plt
import numpy as np
from algorithms.adam import Adam
from algorithms.heavy_ball import Heavy_ball
from algorithms.standard_GD import Standard_GD
from algorithms.accelerated_GD_adaptive import Nesterov_acceleration_adaptive
from datasets.mnist.files import mnist_train_X_y, mnist_test_X_y
from test_runner.test_runner_file import Runner
from models.utility import accuracy, to_torch
import models.one_hidden_relu_softmax as model_no_L2

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
w0 = model_no_L2.initial_params(X_train.shape[1], 300, np.max(y_train) + 1)
step_size = 0.01
order = ['Standard', 'Heavy ball', 'NAG', 'Adam']
algs = [Standard_GD, Heavy_ball, Nesterov_acceleration_adaptive, Adam]

test_set = {
    "w0": w0,
    "GD_params": {"step_size": step_size},
    "alg": algs,
    "model": model_no_L2,
    "max_iter": iterations,
    "data_set": (X_train, y_train),
    "epsilon": epsilon,
}

runner = Runner(dic=test_set)
results = runner.get_result()

x_values = [i for i in range(0, iterations)]

for i, alg_name in enumerate(order):
    alg = algs[i]
    result = runner.get_result(alg=alg)[0]
    plt.plot(x_values, result.get_losses_over_time(), label=f"{alg_name}")

plt.legend(loc="lower left")
plt.yscale('log')
save("convergence_non_convex_loss_comparison")

for i, alg_name in enumerate(order):
    alg = algs[i]
    result = runner.get_result(alg=alg)[0]
    plt.plot(x_values, result.get_grad_norms_over_time(), label=f"{alg_name}")

plt.legend(loc="lower left")
plt.yscale('log')
save('convergence_non_convex_grad_norm')


alg = algs[3]
alg_name = order[3]
result = runner.get_result(alg=alg)[0]
print('Train: ', result.get_best_accuracy())
accs = [accuracy(y_test, model_no_L2.predict(w, X_test)) for w in result.get_weights_over_time()]
print('Test: ', np.max(accs))
plt.plot(x_values, accs, label=f"Test accuracy")
plt.plot(x_values, result.get_accuracy_over_time(), label=f"Train accuracy")
plt.legend(loc="lower right")

save('non_convex_accuracy_no_L2')

alg = algs[0]
result = runner.get_result(alg=alg)[0]
grads = result.get_grad_norms_over_time()
# import pyperclip
# pyperclip.copy('\n'.join([str(g) for g in grads]))
best_loss = runner.get_result(alg=alg)[0].get_losses_over_time()[-1]
start_loss = runner.get_result(alg=alg)[0].get_losses_over_time()[0]
print('best: ', best_loss)
print('start: ', start_loss)
