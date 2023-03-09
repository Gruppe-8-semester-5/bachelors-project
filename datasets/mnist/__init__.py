from datasets.mnist import mnist_files
from datasets.mnist.digit import Digit
if not mnist_files.have_files():
    print("Missing MNIST dataset, fetching it")
    mnist_files.download_files()

