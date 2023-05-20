from typing import Callable

import numpy as np
from analysis.serialization import Serializable

from analysis.utility import euclid_distance

class GradientDescentResult(Serializable):
    """Utility class for gradient descent algorithms. Used for historic values to draw graphs"""
    def __init__(self, 
                 derivation: Callable[[np.ndarray], np.ndarray]) -> None:
        self.diff: Callable[[np.ndarray], np.ndarray] = derivation
        self.weights: list[np.ndarray] = list()
        self.accuracies: list[np.ndarray] = list()
        self.most_accurate_weights_list: list[np.ndarray] = list()
        self.most_accurate_weights = None
        self._derivations_over_time = None

    def add_weight(self, w: np.ndarray):    
        self.weights.append(w)
        if self.most_accurate_weights is not None:
            self.most_accurate_weights_list.append(self.most_accurate_weights)

    def add_accuracy(self, w: np.ndarray):
        self.accuracies.append(w)

    def get_most_accurate_weights(self) -> np.ndarray:
        if self.most_accurate_weights is None:
            raise Exception("No 'best' weight was ever specified. Make sure the algorithm implements this")
        return self.most_accurate_weights
    
    def set_most_accurate_weights(self, weights: np.ndarray):
        self.most_accurate_weights = weights
    
    def set_closest_to_zero_derivation_weight(self, weights: np.ndarray):
        self.closest_to_zero_derivation_weight = weights

    def get_closest_to_zero_derivation_weight(self):
        return self.closest_to_zero_derivation_weight

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
        return self.most_accurate_weights_list
    
    def get_deriviation(self) -> Callable[[np.ndarray], np.ndarray]:
        return self.diff

    def get_distances_to_final_weight(self) -> list[float]:
        final_point = self.get_final_weight()
        return list(map(lambda p: euclid_distance(p, final_point) ,self.get_weights_over_time()))
    
    def get_distances_to_most_accurate_weight(self) -> list[float]:
        most_accurate_point = self.get_most_accurate_weights()
        return list(map(lambda p: euclid_distance(p, most_accurate_point) ,self.get_weights_over_time()))
    
    def get_closest_derivation_distance_to_closest_to_zero_derivation_over_time(self) -> list[float]:
        closest_to_zero_weight = self.get_closest_to_zero_derivation_weight()
        return list(map(lambda p: euclid_distance(p, closest_to_zero_weight) ,self.get_closest_to_zero_derivation_weights_over_time()))
    
    def get_final_distance_to_most_accurate_weight(self) -> float:
        most_accurate_point = self.get_most_accurate_weights()
        return list(map(lambda p: euclid_distance(p, most_accurate_point) ,self.get_weights_over_time()))[len(list(map(lambda p: euclid_distance(p, most_accurate_point) ,self.get_weights_over_time()))) - 1]

    def get_best_weight_over_time_distances_to_best_weight(self) -> list[float]:
        """This function returns the distance to the best weight, from each 'best-so-far' weight of the iteration.
           Using this eliminates the 'bad' steps"""
        most_accurate_point = self.get_most_accurate_weights()
        return list(map(lambda p: euclid_distance(p, most_accurate_point) ,self.get_best_weights_over_time()))

    def get_running_accuracy_average(self, average_size = 20) -> list[float]:
        points = self.get_accuracy_over_time()

        averages = list()
        for i in range(average_size, len(points)):
            averages.append(np.mean(points[i-average_size:i]))

        return averages

    def get_running_distance_to_most_accurate_weights_average(self, average_size = 20) -> list[float]:
        points = self.get_distances_to_most_accurate_weight()
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
    
    def get_derivations_over_time(self) -> np.ndarray:
        weights = self.get_weights_over_time()
        if self._derivations_over_time is None:
            self._derivations_over_time = np.array(list(map(lambda w: self.get_deriviation()(w), weights))) 
        return self._derivations_over_time
    
    def get_closest_to_zero_derivations_over_time(self, from_iteration = 0, to_iteration = None) -> np.ndarray:
        if to_iteration is None:
            weights = self.get_weights_over_time()[from_iteration:]
        else:
            weights = self.get_weights_over_time()[from_iteration : to_iteration]

        best_derivations = []
        best_div = self.get_deriviation()(weights[0])

        for weight in weights:
            zero_distance_to_best = euclid_distance(np.zeros_like(weight), best_div)
            current_div = self.get_deriviation()(weight)
            zero_distance_to_current = euclid_distance(np.zeros_like(weight), current_div)

            if (zero_distance_to_best > zero_distance_to_current):
                best_div = current_div

            best_derivations.append(best_div)

        return best_derivations
    
    def get_closest_to_zero_derivation_weights_over_time(self) -> np.ndarray:
        weights = self.get_weights_over_time()

        best_div = self.get_deriviation()(weights[0])
        closest_weight = weights[0]
        closest_derivation_weights = []

        for weight in weights:
            zero_distance_to_best = euclid_distance(np.zeros_like(weight), best_div)
            current_div = self.get_deriviation()(weight)
            zero_distance_to_current = euclid_distance(np.zeros_like(weight), current_div)

            if (zero_distance_to_best > zero_distance_to_current):
                best_div = current_div
                closest_weight = weight

            closest_derivation_weights.append(closest_weight)

        return closest_derivation_weights
    
    def get_final_derivation(self):
        return self.get_deriviation()(self.get_weights_over_time()[len(self.get_weights_over_time()) - 1])
    
    def get_first_weight(self):
        return self.weights[0]

    def get_derivation_distances_to_zero_over_time(self):
        return list(map(lambda div: euclid_distance(np.zeros_like(div), div), self.get_derivations_over_time()))
    
    def get_best_weight_derivation_distances_to_zero_over_time(self, from_iteration = 0, to_iteration = None):
        return list(map(lambda div: euclid_distance(np.zeros_like(div), div), self.get_closest_to_zero_derivations_over_time(from_iteration, to_iteration)))

    def get_distance_to_best_improvement_deltas(self, allow_zeros=True) -> np.ndarray:
        points = self.get_distances_to_most_accurate_weight()
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
            "best_weights_list": self.most_accurate_weights_list,
            "best_weights": self.most_accurate_weights,
        }
    
    def from_serialized(self, serialized):
        result = GradientDescentResult(self.diff)
        result.weights = serialized["weights"]
        result.accuracies = serialized["accuracies"]
        result.most_accurate_weights_list = serialized["best_weights_list"]
        result.most_accurate_weights = serialized["best_weights"]
        return result