from PIL import Image
import numpy as np
import os.path


class Digit:
    def __init__(self, raw):
        self.raw: str = raw[0]
        raw_split = self.raw.split(",")
        self.label = int(raw_split[0])
        self.features = np.array(list(map(lambda x: int(x), raw_split[1:])))

    def __str__(self) -> str:
        return "label: " + str(self.label) + "\nfeatures: " + str(self.get_reshaped())

    def get_reshaped(self):
        size = self.features.shape[0]
        assert size == 28 * 28
        return np.reshape(self.features, (28, 28))  # Image pictures are 28x28 pixels

    def get_image(self) -> Image:
        """Creates an image of the digit"""
        image_pixels_values = np.uint8(self.get_reshaped())
        return Image.fromarray(image_pixels_values)

    def save_image(self, file_name: str) -> Image:
        """Saves an image to the .out/ directory, overwriting existing file. Only support png"""
        assert not file_name.endswith(".jpeg")
        assert not file_name.endswith(".jpg")
        target_path = ".out/" + file_name
        if os.path.exists(target_path):
            # Overwrites existing files
            os.remove(target_path)
        image_pixels_values = np.uint8(self.get_reshaped())

        self.get_image().save(target_path)
