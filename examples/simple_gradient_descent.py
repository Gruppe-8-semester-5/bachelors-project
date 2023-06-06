"""
Just for documentation
This is an example and will not work if run inside the examples folder.
Either copy the code from this or move it outside the folder.
"""
from matplotlib import pyplot as plt
import numpy as np
from algorithms import gradient_descent_template, standard_GD
from algorithms import GradientDescentResult
def f(w):
    x = w[0]
    y = w[1]
    return pow(x,2) + x * y + pow(y, 2)


def fdiff(w):
    x = w[0]
    y = w[1]
    return np.array([2*x + y, x + 2*y])

def main():
    # initial value
    print("f(x, y) = x^2 + xy + y^2")
    print(f"f(2,3) = {f(np.array([2,3]))}")
    print(f"f'(2,3) = {fdiff(np.array([2,3]))}")
    
    print("Running gradient descent")
    descent_result: GradientDescentResult = gradient_descent_template.find_minima(
        np.array([800,800]), 
        standard_GD.Standard_GD(0.01),
        fdiff, 
        max_iter=200000,
        epsilon=1.0e-2)
    
    minima = descent_result.get_final_weight()
    min_x = minima[0]
    min_y = minima[1]
    no_of_iterations = descent_result.number_of_weights()
    print(f"Found the point ({min_x}, {min_y}) after {no_of_iterations} iterations")
    print(f"f({min_x}, {min_y}) = {f(np.array([min_x, min_y]))}")
    print(f"f'({min_x}, {min_y}) = {fdiff(np.array([min_x, min_y]))}")

    draw_result(descent_result)



def fplot(x, y):
    return pow(x,2) + x * y + pow(y, 2)

def draw_result(result: GradientDescentResult):
    x = np.outer(np.linspace(-400, 800, 20), np.ones(20))
    y = x.copy().T
    z = fplot(x, y)

    fig = plt.figure()
    
    # syntax for 3-D plotting
    ax = plt.axes(projection ='3d')
    
    # syntax for plotting
    i = 0
    for point in result.get_weights_over_time():
        if i < 1800 and i % 8 == 0:
            ax.plot(point[0], point[1], fplot(point[0], point[1]),"r.", zorder=10)
        i+=1

    ax.plot_surface(x, y, z, cmap ='viridis', edgecolor ='green')

    ax.set_title('Surface plot geeks for geeks')
    plt.show()

if __name__ == "__main__":
    main()
