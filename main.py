from matplotlib import pyplot as plt
import numpy as np
from algorithms import simple_gradient_descent
from algorithms import GradientDescentResult
def f(w):
    x = w[0]
    y = w[1]
    return 3*pow(x,2) + 8*y + 2*x*y


def fdiff(w):
    x = w[0]
    y = w[1]
    return np.array([6*x + 2*y, 2*x + 8])

def main():
    # initial value
    print(f"f(2,3,5) = {f(np.array([2,3,5]))}")
    print(f"f'(2,3,5) = {fdiff(np.array([2,3,5]))}")
    
    print("Running gradient descent")
    descent_result: GradientDescentResult = simple_gradient_descent.find_minima(
        np.array([100,40]), 
        0.003, 
        f, 
        fdiff, 
        max_iter=200000,
        epsilon=1.0e-2)
    
    minima = descent_result.get_final_point()
    min_x = minima[0]
    min_y = minima[1]
    no_of_iterations = descent_result.number_of_points()
    print(f"Found the point ({min_x}, {min_y}) after {no_of_iterations} iterations")
    print(f"f({min_x}, {min_y}) = {f(np.array([min_x, min_y]))}")
    print(f"f'({min_x}, {min_y}) = {fdiff(np.array([min_x, min_y]))}")

    draw_result(descent_result)



def fplot(x, y):
    return 3*pow(x,2) + 8*y + 2*x*y

def test(x, y):
    return x + y

def draw_result(result: GradientDescentResult):
    x = np.outer(np.linspace(-150, 150, 20), np.ones(20))
    y = x.copy().T
    z = fplot(x, y)

    fig = plt.figure()
    
    # syntax for 3-D plotting
    ax = plt.axes(projection ='3d')
    
    # syntax for plotting
    i = 0
    for point in result.get_points():
        if i < 1800 and i % 4 == 0:
            ax.plot(point[0], point[1], fplot(point[0], point[1]),"r.", zorder=10)
        i+=1

    ax.plot_surface(x, y, z, cmap ='viridis', edgecolor ='green')

    ax.set_title('Surface plot geeks for geeks')
    plt.show()

if __name__ == "__main__":
    main()
