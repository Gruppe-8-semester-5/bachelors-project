from PIL import Image
import numpy as np
import os.path


class Digit:
    def __init__(self, dataset_row):
        features_split = list(map(lambda x: int(x), dataset_row[1:])) # Converts strings to int
        label_split = dataset_row[0]
        self.label = label_split
        self.raw: str = dataset_row[0]
        self.features = np.array(list(features_split))


    def __str__(self) -> str:
        return f"label: {self.label}\nfeatures:{str(self.get_reshaped_features())}"


    def get_reshaped_features(self):
        """Converts the feature vector to a 28x28 matrix"""
        size = self.features.shape[0]
        assert size == 28 * 28

        # Image pictures are 28x28 pixels
        return np.reshape(self.features, (1, 28, 28))
    
    def get_features(self) -> np.ndarray:
        return self.features
    
    def get_label(self) -> str:
        return self.label
    
    def get_label_int(self) -> int:
        return int(self.label)

    def get_image(self) -> Image:
        """Creates an image of the digit"""
        image_pixels_values = np.uint8(self.get_reshaped_features())
        return Image.fromarray(image_pixels_values)


    def save_image(self, file_name: str) -> Image:
        """ Saves an image to the .out/ directory, overwriting existing file.
            Only support png"""
        if file_name.startswith("/"):
            file_name = file_name[1:]
        target_path = f".out/{file_name}.png"

        if not os.path.exists(".out/"):
            os.mkdir(".out")

        if os.path.exists(target_path):
            # Overwrites existing files
            os.remove(target_path)

        self.get_image().save(target_path)
