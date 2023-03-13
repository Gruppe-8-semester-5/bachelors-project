from typing import Callable

import numpy as np


class GradientDescentResult:
    """Utility class for gradient descent algorithms. Used for historic values to draw graphs"""
    def __init__(self, 
                 function: Callable[[np.ndarray], np.ndarray], 
                 derivation: Callable[[np.ndarray], np.ndarray]) -> None:
        self.f: Callable[[np.ndarray], np.ndarray] = function
        self.diff: Callable[[np.ndarray], np.ndarray] = derivation
        self.points: list[np.ndarray] = list()
    
    def add_point(self, w: np.ndarray):
        self.points.append(w)

    def number_of_points(self):
        return len(self.points)
    
    def get_points(self):
        if self.number_of_points() == 0:
            raise Exception("No points available")
        return self.points
    
    def get_points_axis(self, axis: int):
        points = self.get_points()
        result = list()
        for point in points:
            result.append(point[axis])       
        return result         

    def get_final_point(self) -> np.ndarray:
        if self.number_of_points == 0:
            
            raise Exception("No points available")
        return self.points[self.number_of_points() - 1]