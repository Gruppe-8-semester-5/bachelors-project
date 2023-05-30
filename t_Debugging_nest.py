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
algs = Nesterov_acceleration_adaptive

test_set = {
    "w0": w0,
    "GD_params": {"step_size": step_size},
    "alg": algs,
    "model": model_no_L2,
    "max_iter": iterations,
    "data_set": (X_train, y_train),
    "epsilon": epsilon,
    "batch": None,
}

runner = Runner(dic=test_set)
results = runner.get_result()

x_values = [i for i in range(0, iterations)]

result = runner.get_result()[0]
plt.plot(x_values[460:500], result.get_losses_over_time()[460:500], label=f"Nesterov")
plt.yscale('log')
plt.show()

