from datasets import mnist


def main():
    test_reader = mnist.mnist_files.read_test_data()
    # Grab first item of test set
    sample_letter = test_reader[0]
    letter = mnist.Letter(sample_letter)
    print(letter)

if __name__ == "__main__":
    main()
