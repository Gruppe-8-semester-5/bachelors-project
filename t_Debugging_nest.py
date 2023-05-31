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
weights = result.get_weights_over_time()
import models.logistic_L2 as dumb
x = np.reshape(np.arange(9), (3,3)).astype(np.float64)
y = np.array([0, 1, 0])
d = lambda w: dumb.gradient(x, y, w)
w = np.array([1,2,3]).astype(np.float64)

nest = Nesterov_acceleration_adaptive(step_size = step_size)
for i in range(0, 483):
    nest.step(w, d)
nest.w = weights[482]
nest._prev_w = weights[481] 
nest.moment = nest.get_momentum_term()
d = lambda w: model_no_L2.gradient(X_train, y_train, w)
next = nest.step(weights[482], d)
print(len(np.argwhere(next[1-1] != weights[483][1-1])))
print(len(np.argwhere(next[2-1] != weights[483][2-1])))
print(len(np.argwhere(next[3-1] != weights[483][3-1])))
print(len(np.argwhere(next[4-1] != weights[483][4-1])))
# print(weights[484][0])
# print(np.max(np.abs(weights[484][0])))
# print(np.max(np.abs(weights[484][1])))
# print(np.max(np.abs(weights[484][2])))
# print(np.max(np.abs(weights[484][3])))
# print(np.max(np.abs(next[0])))
# print(np.max(np.abs(next[1])))
# print(np.max(np.abs(next[2])))
# print(np.max(np.abs(next[3])))
result = runner.get_result(alg=algs)[0]
plt.plot(x_values, result.get_losses_over_time(), label=f"Nesterov")
plt.yscale('log')
# plt.xscale('log')
plt.show()
