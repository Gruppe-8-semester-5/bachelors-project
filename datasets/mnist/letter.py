import numpy as np


class Letter:
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
