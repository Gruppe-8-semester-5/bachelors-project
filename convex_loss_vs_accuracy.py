from matplotlib import pyplot as plt
import numpy as np
from datasets.winequality.files import wine_X_y
from models.utility import to_torch
import models.logistic_torch as logistic
import models.softmax_regression as softmax
from models.utility import accuracy as acc

epsilon = 1.0e-11
iterations = 1000

folder = "figures/"


def save(fname):
    plt.savefig(fname=folder + fname + ".png", format="png")
    plt.close()


X, y = wine_X_y()
n = X.shape[0]

np.random.seed(0)

points = np.random.uniform(-10, 10, (100000, *logistic.initial_params(X).shape))

loss = []
accuracy = []
for i in range(points.shape[0]):
    w = points[i]
    loss.append(logistic.negative_log_likelihood(*to_torch(X, y, w)))
    accuracy.append(acc(y, logistic.predict(w, X)))

plt.plot(loss, accuracy, label="Some text", marker="o", markersize=0.2, linestyle="")
plt.xlabel("loss")
plt.ylabel("accuracy")
plt.yscale("log")
plt.xscale("log")
save("loss_vs_accuracy_logistic")


np.random.seed(0)
points = np.random.uniform(-10, 10, (100000, *softmax.initial_params(X, y).shape))

loss = []
accuracy = []
for i in range(points.shape[0]):
    w = points[i]
    loss.append(softmax.loss(*to_torch(X, y, w)))
    accuracy.append(acc(y, softmax.predict(w, X)))

plt.plot(loss, accuracy, label="Some text", marker="o", markersize=0.2, linestyle="")
plt.xlabel("loss")
plt.ylabel("accuracy")
# plt.yscale('log')
# plt.xscale('log')
save("loss_vs_accuracy_softmax")
