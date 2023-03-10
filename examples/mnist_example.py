"""
Just for documentation
This is an example and will not work if run inside the examples folder.
Either copy the code from this or move it outside the folder.
"""
from PIL import Image
import os
from datasets import mnist


def main():
    test_data = mnist.files.read_test_data()    # 10k images
    _ = mnist.files.read_train_data()           # 60k images
    # Grab first item of test set
    sample_letter = test_data[0]
    digit: mnist.Digit = mnist.Digit(sample_letter)
    print(digit)
    making_images(digit)
    export_all_test_images()
    
    features, labels = mnist.data_to_array(test_data)
    i = 0
    for x in features[0]:
        if digit.get_features()[i] != x:
            print("something not right here")
        i += 1

def export_all_test_images():
    if not os.path.exists("./.out/test_digits/"):
        os.mkdir("./.out/test_digits/")
    test_data = mnist.files.read_test_data()

    # Converts test data to digit objects
    test_digits = mnist.data_to_digits(test_data)
    # Saves an image for each test data digit (about 3mb)
    i = 0
    for test_digit in test_digits:
        test_digit.save_image(f"/test_digits/digit_{test_digit.label}_{i}")
        i += 1


def making_images(digit: mnist.Digit):
    # Print letter to a png
    digit.save_image("test")  # Saves image to ./.out/test.png

    # Or we can show it directly without saving it
    image: Image = digit.get_image()
    image.show()


if __name__ == "__main__":
    main()
