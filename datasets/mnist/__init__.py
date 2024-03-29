from datasets.mnist import files
from datasets.mnist.digit import Digit
from datasets.mnist.data import data_to_array
from datasets.mnist.data import data_to_digits

# Automatically downloads files if not present
if not files.have_files():
    print("Missing MNIST dataset, downloading them")
    files.download_files()

