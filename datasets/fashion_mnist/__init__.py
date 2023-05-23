from datasets.fashion_mnist import files

# Automatically downloads files if not present
if not files.have_files():
    print("Missing fashion MNIST dataset, downloading them")
    files.download_files()

