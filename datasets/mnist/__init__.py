from datasets.mnist import files

# Automatically downloads files if not present
if not files.have_files():
    print("Missing MNIST dataset, downloading them")
    files.download_files()

