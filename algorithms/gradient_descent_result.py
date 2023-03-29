from typing import Callable

import numpy as np

from analysis.utility import euclid_distance


class GradientDescentResult:
    """Utility class for gradient descent algorithms. Used for historic values to draw graphs"""
    def __init__(self, 
                 derivation: Callable[[np.ndarray], np.ndarray]) -> None:
        self.diff: Callable[[np.ndarray], np.ndarray] = derivation
        self.points: list[np.ndarray] = list()
        self.accuracies: list[np.ndarray] = list()
    
    def add_point(self, w: np.ndarray):
        self.points.append(w)

    def add_accuracy_point(self, w: np.ndarray):
        self.accuracies.append(w)

    def get_best_weights(self) -> np.ndarray:
        if self.best_weights is None:
            raise Exception("No 'best' weight was ever specified. Make sure the algorithm implements this")
        return self.best_weights
    
    def set_best_weights(self, weights: np.ndarray):
        self.best_weights = weights

    def number_of_points(self):
        return len(self.points)
    
    def get_points(self):
        if self.number_of_points() == 0:
            raise Exception("No points available")
        return self.points
    
    def get_accuracies(self):
        if self.number_of_points() == 0:
            raise Exception("No points available")
        return self.accuracies
    
    def get_points_axis(self, axis: int):
        points = self.get_points()
        result = list()
        for point in points:
            result.append(point[axis])       
        return result
    
    def get_distances_to_final_point(self) -> list[float]:
        final_point = self.get_final_gradient()
        return list(map(lambda p: euclid_distance(p, final_point) ,self.get_points()))
    
    def get_distances_to_best_point(self) -> list[float]:
        best_point = self.get_best_weights()
        return list(map(lambda p: euclid_distance(p, best_point) ,self.get_points()))

    def get_final_gradient(self) -> np.ndarray:
        if self.number_of_points() == 0:
            raise Exception("No points available")
        return self.points[self.number_of_points() - 1]