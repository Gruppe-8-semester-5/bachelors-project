import csv
import os
from typing import Tuple
import numpy as np

from datasets.mnist.digit import Digit

from datasets.utility.download import download

DOWNLOAD_DESTINATION = ".data/"

def download_files():
    download("http://95.217.87.122:4321/datasets/mnist_test.csv", DOWNLOAD_DESTINATION)
    download("http://95.217.87.122:4321/datasets/mnist_train.csv", DOWNLOAD_DESTINATION)
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

def mnist_train_X_y(num_elements:int = None) -> Tuple[np.ndarray, np.ndarray]:
    if num_elements is None:
        dataset = read_train_data()
    else:
        dataset = read_train_data()
        np.random.shuffle(dataset)
        dataset = dataset[0:num_elements]
    digits: list[Digit] = list(map(lambda d: Digit(d), dataset))
    color_label_list = list(map(lambda digit: digit.get_label_int(), digits))
    y = np.array(color_label_list)
    normalizer = lambda digit: (digit.get_features() - np.mean(digit.get_features()))/np.std(digit.get_features()) # (x - mean(x))/std(x)
    feature_list = list(map(normalizer, digits))
    X = np.array(feature_list).astype(float)
    return X, y

def mnist_test_X_y(num_elements:int = None) -> Tuple[np.ndarray, np.ndarray]:
    if num_elements is None:
        dataset = read_test_data()
    else:
        dataset = read_test_data()
        np.random.shuffle(dataset)
        dataset = dataset[0:num_elements]
    digits: list[Digit] = list(map(lambda d: Digit(d), dataset))
    color_label_list = list(map(lambda digit: digit.get_label_int(), digits))
    y = np.array(color_label_list)
    normalizer = lambda digit: (digit.get_features() - np.mean(digit.get_features()))/np.std(digit.get_features()) # (x - mean(x))/std(x)
    feature_list = list(map(normalizer, digits))
    X = np.array(feature_list).astype(float)
    return X, y

def mnist_X_y_simpel():
    dataset = read_train_data()
    digits: list[Digit] = list(map(lambda d: Digit(d), dataset))
    color_label_list = list(map(lambda digit: digit.get_label_int(), digits))
    y = np.array(color_label_list)
    normalizer = lambda digit: (digit.get_features() - np.mean(digit.get_features()))/np.std(digit.get_features()) # (x - mean(x))/std(x)
    feature_list = list(map(normalizer, digits))
    X = np.array(feature_list).astype(float)
    return X, y


def mnist_X_y():
    dataset = read_train_data()
    digits: list[Digit] = list(map(lambda d: Digit(d), dataset))
    color_label_list = list(map(lambda digit: digit.get_label_int(), digits))
    y = np.array(color_label_list)
    normalizer = lambda digit: (digit.get_reshaped_features() - np.mean(digit.get_reshaped_features()))/np.std(digit.get_reshaped_features()) # (x - mean(x))/std(x)
    feature_list = list(map(normalizer, digits))
    X = np.array(feature_list).astype(float)
    return X, y
