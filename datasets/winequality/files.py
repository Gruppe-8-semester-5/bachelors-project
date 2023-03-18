import csv
import os

from datasets.utility.download import download

DOWNLOAD_DESTINATION = ".data/"

def download_files():
    download("http://dlsailaway.tk:4321/datasets/wine_quality.csv", DOWNLOAD_DESTINATION)
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
