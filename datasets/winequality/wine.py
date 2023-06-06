import numpy as np


class Wine:
    def __init__(self, dataset_row):
        self.fixed_acidity = float(dataset_row[0])
        self.volatile_acidity = float(dataset_row[1])
        self.citric_acid = float(dataset_row[2])
        self.residual_sugar = float(dataset_row[3])
        self.chlorides = float(dataset_row[4])
        self.free_sulfur_dioxide = float(dataset_row[5])
        self.total_sulfur_dioxide = float(dataset_row[6])
        self.density = float(dataset_row[7])
        self.pH = float(dataset_row[8])
        self.sulphates = float(dataset_row[9])
        self.alcohol = float(dataset_row[10])
        self.quality = int(dataset_row[11])
        self.color = dataset_row[12]

    def get_feature_vector(self) -> np.ndarray:
        return np.array([
            self.fixed_acidity,
            self.volatile_acidity,
            self.citric_acid,
            self.residual_sugar,
            self.chlorides,
            self.free_sulfur_dioxide,
            self.total_sulfur_dioxide,
            self.density,
            self.pH,
            self.sulphates,
            self.alcohol
        ])
    
    def get_quality(self) -> int:
        return self.quality - 3
    
    def get_color(self)->str:
        return self.color
    
    def __str__(self) -> str:
        return f"{self.color} wine of quality {self.quality}\nfeatures:{str(self.get_feature_vector())}"
