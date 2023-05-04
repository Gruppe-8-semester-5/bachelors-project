from matplotlib import pyplot as plt
import numpy as np
from algorithms.gradient_descent_result import GradientDescentResult


class GradientDescentResultPlotter:
    
    results: list[GradientDescentResult]
    labels: list[str]

    def __init__(self, results: list[GradientDescentResult], labels: list[str] = []) -> None:
        # Check all results are of the same dimension
        if len(results) < 1:
            raise Exception("There must be at least 1 GradientDescentResult for plotter!")
        first_result = results[0]
        self.labels = labels
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
    
    def get_linear_x_values(self, y_values_for_plot: np.ndarray | list[float]):
        x_values = [x for x in range(0, len(y_values_for_plot.get_derivation_distances_to_zero_over_time()))]
        return x_values

    def plot_accuracies_over_time(self):
        x_values = self.get_linear_x_values(self.results[0].get_accuracy_over_time())
        
        for index, result in enumerate(self.results):
            label = self.labels[index]
            plt.plot(x_values, result.get_accuracy_over_time(), label=label)

        return self
    
    def plot(self, legend: None | str = None):
        if legend is not None:
            plt.legend(loc='center right')
        plt.show()