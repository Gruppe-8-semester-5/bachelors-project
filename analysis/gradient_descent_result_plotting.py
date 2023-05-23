from typing import Callable
from matplotlib import pyplot as plt
import numpy as np
from algorithms.gradient_descent_result import GradientDescentResult


class GradientDescentResultPlotter:
    """Builder pattern for showing plots
    Ends with `.plot()`

    Example use
    `GradientDescentResultPlotter(results).plot_accuracies_over_time().plot()`
    """

    _results: list[GradientDescentResult]
    _result_labels: list[str]
    _function_labels: list[str]
    _legend_placement: str | None
    _y_axis_hidden: bool
    _x_values: list[int | float] | np.ndarray | None
    _y_logarithmic: bool
    _y_limit: None | tuple

    def __init__(self, results: list[GradientDescentResult]) -> None:
        # Check all results are of the same dimension
        if len(results) < 1:
            raise Exception(
                "There must be at least 1 GradientDescentResult for plotter!"
            )
        first_result = results[0]
        self._result_labels = []
        self._function_labels = []
        self._plot_targets = []
        self._plotted_functions = []
        self._legend_placement = None
        self._results = results
        self._y_axis_hidden = False
        self._x_values = None
        self._y_logarithmic = False
        self._y_limit = None

        for result in results:
            if len(result.get_accuracy_over_time()) != len(
                first_result.get_accuracy_over_time()
            ):
                raise Exception(
                    "Failed to create plotter. All results must have comparable data. Not all accuracy arrays are of the same size!"
                )
            if len(result.get_weights_over_time()) != len(
                first_result.get_weights_over_time()
            ):
                raise Exception(
                    "Failed to create plotter. All results must have comparable data. Not all weight arrays are of the same size!"
                )
            if result.get_final_weight().shape != first_result.get_final_weight().shape:
                print(
                    "Warning! All results do not have comparable data. Weight shape are not the same! Some plotting functions will not work properly."
                )

        # Initialize plt to plot
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True

    def _get_linear_x_values(self, y_values_for_plot: np.ndarray | list[float]):
        x_values = [x for x in range(0, len(y_values_for_plot))]
        return x_values

    def plot_with_y_axis_logarithmic(self):
        self._y_logarithmic = True
        return self

    def plot_accuracies_over_time(self, plot_from=0, plot_to=None):
        if plot_to is None:
            return self._add_plot_values(
                lambda gd_result: gd_result.get_accuracy_over_time()[plot_from:]
            )
        return self._add_plot_values(
            lambda gd_result: gd_result.get_accuracy_over_time()[plot_from:plot_to]
        )

    def plot_distance_to_final_weight_over_time(self):
        return self._add_plot_values(
            lambda gd_result: gd_result.get_distances_to_final_weight()
        )

    def plot_distance_to_zero_gradient_over_time(self):
        return self._add_plot_values(
            lambda gd_result: gd_result.get_derivation_distances_to_zero_over_time()
        )

    def plot_best_weight_distance_to_zero_gradient_over_time(
        self, plot_from=0, plot_to=None
    ):
        if plot_to is None:
            return self._add_plot_values(
                lambda gd_result: gd_result.get_best_weight_derivation_distances_to_zero_over_time(
                    plot_from
                )
            )
        return self._add_plot_values(
            lambda gd_result: gd_result.get_best_weight_derivation_distances_to_zero_over_time(
                plot_from, plot_to
            )
        )

    def with_x_values(self, x_values):
        self._x_values = x_values
        return self

    def plot_function(self, function: Callable[[int], float]):
        self._plotted_functions.append(function)
        return self

    def plot_distance_to_absolute_most_accurate_weight(self):
        best_performer = self._results[0]
        # Find best performer
        for result in self._results:
            if best_performer.get_best_accuracy() < result.get_best_accuracy():
                best_performer = result

        # Set best weights for other results
        for result in self._results:
            result.set_most_accurate_weights(best_performer.get_most_accurate_weights())

        return self._add_plot_values(
            lambda gd_result: gd_result.get_distances_to_most_accurate_weight()
        )

    def _add_plot_values(
        self, plot_target_func: Callable[[GradientDescentResult], np.ndarray]
    ):
        if self._x_values is None:
            self._x_values = self._get_linear_x_values(
                plot_target_func(self._results[0])
            )

        for result in self._results:
            self._plot_targets.append(plot_target_func(result))
        return self

    def legend_placed(self, location: str = "center right"):
        self._legend_placement = location
        return self

    def set_y_limit(self, min: float, max: float):
        self._y_limit = (min, max)
        return self

    def hide_y_axis(self):
        self._y_axis_hidden = True
        return self

    def with_result_labelled(self, labels: list[str] = []):
        self._result_labels = labels
        return self

    def with_functions_labelled(self, labels: list[str] = []):
        self._function_labels = labels
        return self

    def plot_loss(self):
        self._add_plot_values(lambda gd_result: gd_result.get_losses_over_time())
        return self

    def plot_optimal_loss_difference(self, target_loss: float):
        """
        Plots f(w_i) - f(w*)
        Takes target_loss as argument, i.e f(w*)
        """
        self._add_plot_values(
            lambda gd_result: gd_result.get_losses_over_time() - target_loss
        )
        return self

    def plot(self):
        if self._x_values is None:
            raise Exception("Failed to find any x-values for plot!")

        if self._y_logarithmic:
            plt.yscale("log")

        for index, target in enumerate(self._plot_targets):
            label = (
                self._result_labels[index] if len(self._result_labels) > index else None
            )
            print(label)
            plt.plot(self._x_values, target, label=label)
        for index, target in enumerate(self._plotted_functions):
            label = (
                self._function_labels[index]
                if len(self._function_labels) > index
                else None
            )
            print(label)
            plt.plot(self._x_values, [target(x) for x in self._x_values], label=label)
        if self._legend_placement is not None:
            plt.legend(loc=self._legend_placement)

        if self._y_axis_hidden:
            ax = plt.gca()
            ax.get_yaxis().set_visible(False)

        if self._y_limit is not None:
            ax = plt.gca()
            ax.set(ylim=self._y_limit)

        plt.show()
