from datasets.winequality.files import read_wine_data
from datasets.winequality.files import read_wine_headers
from datasets.mnist.files import read_mnist_headers
from datasets.winequality.wine import Wine
def main():
    wine_headers = read_wine_headers()
    mnist_headers = read_mnist_headers()
    wine_data = read_wine_data()
    print(wine_headers)
    wine = Wine(wine_data[0])
    print(wine)
    # print(wine_data)
    
if __name__ == "__main__":
    main()
 