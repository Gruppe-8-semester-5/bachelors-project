import csv
import os
from typing import Tuple
import numpy as np
import torch
from torchvision import datasets, transforms

DOWNLOAD_DESTINATION = ".data/"


def download_files():
    datasets.FashionMNIST(DOWNLOAD_DESTINATION, download=True, train=True)
    datasets.FashionMNIST(DOWNLOAD_DESTINATION, download=True, train=False)
    print("Downloaded fashion MNIST")

def have_files() -> bool:
    return os.path.exists(DOWNLOAD_DESTINATION + "FashionMNIST/raw/t10k-images-idx3-ubyte")

def fashion_mnist_X_y() -> Tuple[np.ndarray, np.ndarray]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,),)])
    trainset = datasets.FashionMNIST(DOWNLOAD_DESTINATION, download=True, train=True, transform = transform)
    testset = datasets.FashionMNIST(DOWNLOAD_DESTINATION, download=True, train=False, transform = transform)
    X = []
    y = []
    for set in [trainset, testset]:
        x = torch.utils.data.DataLoader(set)
        dataiter = iter(x)
        for img, label in dataiter:
            X.append(torch.flatten(img).tolist())
            y.append(label.item())
    return np.array(X), np.array(y)
