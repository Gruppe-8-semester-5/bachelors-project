import csv
import os
from typing import Tuple
import numpy as np

import torch
from datasets.mnist.digit import Digit

from datasets.utility.download import download

DOWNLOAD_DESTINATION = ".data/"

def download_files():
    download("http://dlsailaway.tk:4321/datasets/mnist_test.csv", DOWNLOAD_DESTINATION)
    download("http://dlsailaway.tk:4321/datasets/mnist_train.csv", DOWNLOAD_DESTINATION)
    print("Downloaded MNIST dataset")


def have_files() -> bool:
    return os.path.exists(DOWNLOAD_DESTINATION + "mnist_test.csv") \
           and os.path.exists(DOWNLOAD_DESTINATION + "mnist_train.csv")


def read_test_data():
    result = []
    with open(DOWNLOAD_DESTINATION + "./mnist_test.csv", 'r') as file:
        csvreader = csv.reader(file, delimiter=':')
        for row in csvreader:
            row_data = row[0].split(",")
            result.append(row_data)
    return result[1:] # Skip the first, because its the header labels of a table


def read_mnist_headers():
    result = []
    with open(DOWNLOAD_DESTINATION + "./mnist_test.csv", 'r') as file:
        csvreader = csv.reader(file, delimiter=':')
        for row in csvreader:
            result.append(row)
    return str(result[0][0]).split(",")


def read_train_data():
    result = []
    with open(DOWNLOAD_DESTINATION + "./mnist_train.csv", 'r') as file:
        csvreader = csv.reader(file, delimiter=':')
        for row in csvreader:
            row_data = row[0].split(",")
            result.append(row_data)
    return result[1:] # Skip the first, because its the header labels of a table

def mnist_train_X_y(one_in_k = True) -> Tuple[list, list]:
    dataset = read_train_data()
    digits: list[Digit] = list(map(lambda d: Digit(d), dataset))
    color_label_list = list(map(lambda digit: digit.get_label_int(), digits))
    y = np.array(color_label_list)
    if one_in_k:
        # Apparently one_hot must have a int64, but numpy int-arrays translates to int32.
        y = torch.nn.functional.one_hot(torch.from_numpy(y).to(torch.int64)).numpy()
    feature_list = list(map(lambda digit: digit.get_features(), digits))
    X = np.array(feature_list).astype(float)
    return X, y

