import csv
import os
import requests

DOWNLOAD_DESTINATION = ".data/"


def download(url: str, dest_folder: str):
    """Src: https://stackoverflow.com/questions/56950987/download-file-from-url-and-save-it-in-a-folder-python"""
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))


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


def read_train_data():
    result = []
    with open(DOWNLOAD_DESTINATION + "./mnist_train.csv", 'r') as file:
        csvreader = csv.reader(file, delimiter=':')
        for row in csvreader:
            result.append(row)
    return result[1:] # Skip the first, because its the header labels of a table
