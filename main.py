import numpy as np
from datasets.winequality.files import read_wine_data
from datasets.winequality.files import read_wine_headers
from datasets.mnist.files import read_mnist_headers
from datasets.winequality.wine import Wine
def main():
    wine_headers = read_wine_headers()
    mnist_headers = read_mnist_headers()
    wine_data = read_wine_data()
    normsum = 0
    for wine_raw in wine_data:
        wine = Wine(wine_raw)
        normsum += np.linalg.norm(wine.get_feature_vector())
    print(normsum * (1/(2*len(wine_data))))
    print(normsum)
    print(len(wine_data))
if __name__ == "__main__":
    main()
 