from typing import Callable

import numpy as np
from src.analysis.serialization import Serializable

from src.analysis.utility import euclid_distance

class GradientDescentResult(Serializable):
    """Utility class for gradient descent algorithms. Used for historic values to draw graphs"""
    def __init__(self, 
                 derivation: Callable[[np.ndarray], np.ndarray]) -> None:
        self.diff: Callable[[np.ndarray], np.ndarray] = derivation
        self.weights: list[np.ndarray] = list()
        self.accuracies: list[np.ndarray] = list()
        self.best_weights_list: list[np.ndarray] = list()
        self.best_weights = None

    def add_weight(self, w: np.ndarray):
        self.weights.append(w)
        if self.best_weights is not None:
            self.best_weights_list.append(self.best_weights)

    def add_accuracy(self, w: np.ndarray):
        self.accuracies.append(w)

    def get_best_weights(self) -> np.ndarray:
        if self.best_weights is None:
            raise Exception("No 'best' weight was ever specified. Make sure the algorithm implements this")
        return self.best_weights
    
    def set_best_weights(self, weights: np.ndarray):
        self.best_weights = weights

    def number_of_weights(self):
        return len(self.weights)
    
    def get_weights_over_time(self):
        if self.number_of_weights() == 0:
            raise Exception("No points available")
        return self.weights
    
    def get_accuracy_over_time(self):
        if self.number_of_weights() == 0:
            raise Exception("No points available")
        return self.accuracies
    
    def get_weights_axis(self, axis: int):
        points = self.get_weights_over_time()
        result = list()
        for point in points:
            result.append(point[axis])       
        return result
    
    def get_best_weights_over_time(self): 
        return self.best_weights_list
    
    def get_distances_to_final_weight(self) -> list[float]:
        final_point = self.get_final_weight()
        return list(map(lambda p: euclid_distance(p, final_point) ,self.get_weights_over_time()))
    
    def get_distances_to_best_weight(self) -> list[float]:
        best_point = self.get_best_weights()
        return list(map(lambda p: euclid_distance(p, best_point) ,self.get_weights_over_time()))

    def get_best_weight_over_time_distances_to_best_weight(self) -> list[float]:
        """This function returns the distance to the best weight, from each 'best-so-far' weight of the iteration.
           Using this eliminates the 'bad' steps"""
        best_point = self.get_best_weights()
        return list(map(lambda p: euclid_distance(p, best_point) ,self.get_best_weights_over_time()))

    def get_running_accuracy_average(self, average_size = 20) -> list[float]:
        points = self.get_accuracy_over_time()

        averages = list()
        for i in range(average_size, len(points)):
            averages.append(np.mean(points[i-average_size:i]))

        return averages

    def get_running_distance_to_best_weights_average(self, average_size = 20) -> list[float]:
        points = self.get_distances_to_best_weight()
        averages = list()
        for i in range(average_size, len(points)):
            averages.append(np.mean(points[i-average_size:i]))
        return averages

    def get_final_weight(self) -> np.ndarray:
        if self.number_of_weights() == 0:
            raise Exception("No points available")
        return self.weights[self.number_of_weights() - 1]
    
    def get_best_accuracy(self) -> float:
        return np.max(self.accuracies)
    

    def get_distance_to_best_improvement_deltas(self, allow_zeros=True) -> np.ndarray:
        points = self.get_distances_to_best_weight()
        result = list()
        prev_delta = 0
        for i in range(1, len(points)):
            point = points[i]
            prev_point = points[i - 1]
            delta = prev_point - point
            if allow_zeros:
                result.append(delta)
            else:
                result.append(np.max([prev_delta, delta]))
        return np.array(result)


    def to_serialized(self) -> dict:
        return {
            "weights": self.weights,
            "accuracies": self.accuracies,
            "best_weights_list": self.best_weights_list,
            "best_weights": self.best_weights,
        }
    
    def from_serialized(self, serialized):
        result = GradientDescentResult(self.diff)
        result.weights = serialized["weights"]
        result.accuracies = serialized["accuracies"]
        result.best_weights_list = serialized["best_weights_list"]
        result.best_weights = serialized["best_weights"]
        return result