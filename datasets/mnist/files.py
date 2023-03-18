import csv
import os

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
            result.append(row)
    return result[1:] # Skip the first, because its the header labels of a table


def read_headers():
    result = []
    with open(DOWNLOAD_DESTINATION + "./mnist_test.csv", 'r') as file:
        csvreader = csv.reader(file, delimiter=':')
        for row in csvreader:
            result.append(row)
    return result[0] 


def read_train_data():
    result = []
    with open(DOWNLOAD_DESTINATION + "./mnist_train.csv", 'r') as file:
        csvreader = csv.reader(file, delimiter=':')
        for row in csvreader:
            result.append(row)
    return result[1:] # Skip the first, because its the header labels of a table
