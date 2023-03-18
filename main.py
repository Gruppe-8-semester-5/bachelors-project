from datasets.winequality.files import read_wine_data
from datasets.winequality.files import read_wine_headers
def main():
    wine_headers = read_wine_headers()
    wine_data = read_wine_data()
    print(wine_headers)
    # print(wine_data)
    
if __name__ == "__main__":
    main()
