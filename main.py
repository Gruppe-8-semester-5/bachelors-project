from PIL import Image

from datasets import mnist


def main():
    test_reader = mnist.mnist_files.read_test_data()
    # Grab first item of test set
    sample_letter = test_reader[0]
    letter = mnist.Digit(sample_letter)
    print(letter)
    # Print letter to a png
    # letter.save_image("test.png")
    # Or we can show it directly without saving it
    letter.get_image().show()


if __name__ == "__main__":
    main()
