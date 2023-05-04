from typing import Callable
from matplotlib import pyplot as plt
import numpy as np
from algorithms.gradient_descent_result import GradientDescentResult


class GradientDescentResultPlotter:
    
    results: list[GradientDescentResult]
    labels: list[str]
    legend_placement: str | None

    def __init__(self, results: list[GradientDescentResult]) -> None:
        # Check all results are of the same dimension
        if len(results) < 1:
            raise Exception("There must be at least 1 GradientDescentResult for plotter!")
        first_result = results[0]
        self.labels = []
        self.plot_targets = []
        self.legend_placement = None
        self.results = results;
        for result in results:
            if len(result.get_accuracy_over_time()) != len(first_result.get_accuracy_over_time()):
                raise Exception("Failed to create plotter. All results must have comparable data. Not all accuracy arrays are of the same size!")
            if len(result.get_weights_over_time()) != len(first_result.get_weights_over_time()):
                raise Exception("Failed to create plotter. All results must have comparable data. Not all weight arrays are of the same size!")
            if result.get_final_weight().shape != first_result.get_final_weight().shape:
                raise Exception("Failed to create plotter. All results must have comparable data. Weight shape are not the same!")
        
        # Initialize plt to plot 
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
    
    def _get_linear_x_values(self, y_values_for_plot: np.ndarray | list[float]):
        x_values = [x for x in range(0, len(y_values_for_plot))]
        return x_values

    def plot_accuracies_over_time(self):
        return self._add_plot_values(lambda gd_result: gd_result.get_accuracy_over_time())
    
    def plot_distance_to_final_weight_over_time(self):
        return self._add_plot_values(lambda gd_result: gd_result.get_distances_to_final_weight())
    
    def plot_distance_to_absolute_best_weight(self):
        best_performer = self.results[0]
        # Find best performer
        for result in self.results:
            if best_performer.get_best_accuracy() < result.get_best_accuracy():
                best_performer = result

        # Set best weights for other results
        for result in self.results:
            result.set_most_accurate_weights(best_performer.get_most_accurate_weights())

        return self._add_plot_values(lambda gd_result: gd_result.get_distances_to_most_accurate_weight())
    
    def _add_plot_values(self, plot_target_func: Callable[[GradientDescentResult], np.ndarray]):
        self.x_values = self._get_linear_x_values(plot_target_func(self.results[0]))
        
        for result in self.results:
            self.plot_targets.append(plot_target_func(result))
        return self

    def legend_placed(self, location: str = 'center right'):
        self.legend_placement = location
        return self
    
    def with_labels(self, labels: list[str] = []):
        self.labels = labels
        return self

    def plot(self):
        if self.x_values is None:
            raise Exception("Failed to find any x-values for plot!")
        
        for index, target in enumerate(self.plot_targets):
            label = self.labels[index] if len(self.labels) > index else None
            print(label)
            plt.plot(self.x_values, target, label=label)
        if self.legend_placement is not None:
            plt.legend(loc=self.legend_placement)
        plt.show()