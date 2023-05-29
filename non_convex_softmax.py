from matplotlib import pyplot as plt
import numpy as np
from algorithms.adam import Adam
from algorithms.heavy_ball import Heavy_ball
from algorithms.standard_GD import Standard_GD
from algorithms.accelerated_GD_adaptive import Nesterov_acceleration_adaptive
from datasets.mnist.files import mnist_train_X_y, mnist_test_X_y
from test_runner.test_runner_file import Runner
from models.utility import accuracy, to_torch
import models.softmax_regression as softmax

epsilon = 0
iterations = 500

folder = "figures/"


def save(fname):
    plt.savefig(fname=folder + fname + ".png", format="png")
    plt.close()

X_train, y_train = mnist_train_X_y()

X_test, y_test = mnist_test_X_y()
n = X_train.shape[0]

np.random.seed(0)
w0 = softmax.initial_params(X_train, y_train)
step_size = 0.01
order = ['Adam']
algs = [Adam]

test_set = {
    "w0": w0,
    "GD_params": {"step_size": step_size},
    "alg": algs,
    "model": softmax,
    "max_iter": iterations,
    "data_set": (X_train, y_train),
    "epsilon": epsilon,
    "batch": None,
}

runner = Runner(dic=test_set)
results = runner.get_result()

x_values = [i for i in range(0, iterations)]

alg = algs[0]
alg_name = order[0]
result = runner.get_result(alg=alg)[0]
print('Train: ', result.get_best_accuracy())
accs = [accuracy(y_test, softmax.predict(w, X_test)) for w in result.get_weights_over_time()]
print('Test: ', np.max(accs))
plt.plot(x_values, accs, label=f"Test accuracy")
plt.plot(x_values, result.get_accuracy_over_time(), label=f"Train accuracy")
plt.legend(loc="lower right")

# plt.legend(loc="upper right")
# plt.show()
save('non_convex_accuracy_softmax')
