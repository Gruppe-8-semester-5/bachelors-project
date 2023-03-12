import numpy as np
from algorithms import simple_gradient_descent
from algorithms import GradientDescentResult
def f(w):
    x = w[0]
    y = w[1]
    z = w[2]
    return pow(x, 4) + 2 * pow(y,2) + 6 * z

def fdiff(w):
    x = w[0]
    y = w[1]
    return np.array([4 * pow(x, 3), 4*y + 6, 0])

def main():
    # initial value
    print(f"f(2,3,5) = {f(np.array([2,3,5]))}")
    print(f"f'(2,3,5) = {fdiff(np.array([2,3,5]))}")
    
    print("Running gradient descent")
    descent_result: GradientDescentResult = simple_gradient_descent.find_minima(
        np.array([2,3,5]), 
        0.003, 
        f, 
        fdiff, 
        max_iter=2000000,
        epsilon=1.0e-6)
    
    minima = descent_result.get_final_point()
    min_x = minima[0]
    min_y = minima[1]
    min_z = minima[2]
    no_of_iterations = descent_result.number_of_points()
    print(f"Found the point ({min_x}, {min_y}, {min_z}) after {no_of_iterations} iterations")
    print(f"f({min_x}, {min_y}, {min_z}) = {f(np.array([min_x, min_y, min_z]))}")
    print(f"f'({min_x}, {min_y}, {min_z}) = {fdiff(np.array([min_x, min_y, min_z]))}")


if __name__ == "__main__":
    main()
