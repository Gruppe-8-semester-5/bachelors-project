import csv
import os
from typing import Tuple

import numpy as np
import torch

from datasets.utility.download import download
from datasets.winequality.wine import Wine

DOWNLOAD_DESTINATION = ".data/"

def download_files():
    download("http://95.217.87.122:4321/datasets/wine_quality.csv", DOWNLOAD_DESTINATION)
    print("Downloaded wine quality dataset")


def have_files() -> bool:
    return os.path.exists(DOWNLOAD_DESTINATION + "wine_quality.csv")

def read_wine_data():
    result = []
    with open(DOWNLOAD_DESTINATION + "./wine_quality.csv", 'r') as file:
        csvreader = csv.reader(file, delimiter=':')
        for row in csvreader:
            row_data = row[0].split(",")
            result.append(row_data)
    return result[1:] # Skip the first, because its the header labels of a table

def read_wine_headers() -> list:
    result = []
    with open(DOWNLOAD_DESTINATION + "./wine_quality.csv", 'r') as file:
        csvreader = csv.reader(file, delimiter=':')
        for row in csvreader:
            result.append(row)
            break # Read the first line, the break the loop
    return str(result[0][0]).split(",")

def color_to_label(color):
    if color == "white":
        return 0
    return 1


def wine_X_y() -> Tuple[list, list]:
    dataset = read_wine_data()
    wines: list[Wine] = list(map(lambda d: Wine(d), dataset))
    color_label_list = list(map(lambda wine: color_to_label(wine.get_color()), wines))
    y = np.array(color_label_list)
    # Normalized features (centering around origo. [[1],[2]] -> [[-0.5],[0.5]])
    feature_normalizer = lambda wine: (wine.get_feature_vector() -np.mean(wine.get_feature_vector())) /  np.std(wine.get_feature_vector())
    feature_list = list(map(feature_normalizer, wines))
    X = np.array(feature_list)
    return X, y

def wine_X_y_quality_with_color_feature() -> Tuple[list, list]:
    dataset = read_wine_data()
    wines: list[Wine] = list(map(lambda d: Wine(d), dataset))
    color_label_list = list(map(lambda wine: color_to_label(wine.get_color()), wines))
    y = np.array(color_label_list)
    # Normalized features (centering around origo. [[1],[2]] -> [[-0.5],[0.5]])
    feature_normalizer = lambda wine: (wine.get_feature_vector() -np.mean(wine.get_feature_vector())) /  np.std(wine.get_feature_vector())
    feature_list = list(map(feature_normalizer, wines))
    X = np.array(feature_list)
    for i, wine in enumerate(wines):
        np.append(X[i], 0 if wine.get_color() == "white" else 1)

    return X, y

