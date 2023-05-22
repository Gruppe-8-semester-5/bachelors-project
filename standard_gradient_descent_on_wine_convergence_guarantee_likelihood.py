import matplotlib.pyplot as plt
import numpy as np
from algorithms.accelerated_GD import Nesterov_acceleration
from analysis.gradient_descent_result_plotting import GradientDescentResultPlotter
from datasets.winequality.files import read_wine_data, wine_X_y
from analysis.lipschitz import lipschitz_binary_neg_log_likelihood
from datasets.winequality.wine import Wine
from models import logistic_torch
from algorithms import GradientDescentResult, gradient_descent_template, standard_GD
from algorithms.standard_GD import Standard_GD
from algorithms.momentum_GD import Momentum
from models.utility import accuracy, to_torch
from analysis.utility import dump_array_to_csv, euclid_distance
from test_runner.test_runner_file import Runner

# Features and labels
X, y = wine_X_y()

# Lipschitz bound
L = lipschitz_binary_neg_log_likelihood(X, y)

# Starting point
np.random.seed(123) # Set seed for replication purposes
start_gradient = logistic_torch.initial_params(X)

test_set = {
    'w0': start_gradient,
    'GD_params': {'step_size': 1/L},
    'alg': [Standard_GD],
    'model': logistic_torch,
    'max_iter': 100000,
    'data_set': (X, y),
    'epsilon':1.0e-2,
    'auto_stop': False,
    'batch': None
}
runner = Runner(dic = test_set)
descent_result_lipchitz_best = runner.get_result()[0]

# Run GD to test convergence
iterations = 100
test_set = {
    'w0': start_gradient,
    'GD_params': {'step_size': 1/L},
    'alg': [Standard_GD],
    'model': logistic_torch,
    'max_iter': iterations,
    'data_set': (X, y),
    'epsilon':1.0e-2,
    'auto_stop': False,
    'batch': None
}
runner = Runner(dic = test_set)
descent_result_lipchitz = runner.get_result()[0]

# Collect weights and map to NLL values
weights = descent_result_lipchitz.get_weights_over_time()
likelihood_values = [logistic_torch.negative_log_likelihood(*to_torch(X, y, weight)) for weight in weights]
# Find w* (approximate)
w_star = descent_result_lipchitz_best.get_weights_over_time()[-1]
f_w_star = logistic_torch.negative_log_likelihood(*to_torch(X, y, w_star))

# Compute f(w) - f(w*)
diff_to_w_star = [np.abs(nll_val - f_w_star) for nll_val in likelihood_values]

# Discard any 'backwards' steps to only keep lower bound values
best_diff_to_w_star = diff_to_w_star[0]
diff_to_w_star_only_best = [ ]
for current_diff_to_w_star in diff_to_w_star:
    best_diff_to_w_star = min(best_diff_to_w_star, current_diff_to_w_star)
    diff_to_w_star_only_best.append(best_diff_to_w_star)

convergence_rate_multiplier = 1.5 # Labeled C in plot
GradientDescentResultPlotter([descent_result_lipchitz]).plot_function(lambda x: convergence_rate_multiplier/x if x!=0 else convergence_rate_multiplier).plot_function(lambda x: diff_to_w_star[x]).legend_placed("upper right").with_functions_labelled(["C/k", "f(w_k) - f(w*)"]).with_x_values(range(0,iterations)).plot()

#GradientDescentResultPlotter([descent_result_lipchitz]).plot_function(lambda x :-likelihood_values[x] if -likelihood_values[x] > 0 else 0).plot_function(lambda x: convergence_rate_multiplier/x if x != 0 else convergence_rate_multiplier).legend_placed("upper right").with_functions_labelled(["f(w)", "2.750.000/x"]).with_x_values(range(0,iterations)).plot()


#.set_y_limit(0, 0.016*1e6)