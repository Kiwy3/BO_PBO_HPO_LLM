import numpy as np
import matplotlib.pyplot as plt


def function(x):
    print(x)
    #x = x[0] if len(x)==2 else x
    #return {"obj":np.sin(x)**3+np.sqrt(x+5)}
    return np.sin(x)**3+np.sqrt(x+5)

X = np.linspace(-5,5,100)
Y = function(X)

plt.scatter(X,Y)
plt.show()


def soo_1d(f, bounds, num_iterations=100, num_candidates=3):
    """
    Simultaneous Optimistic Optimization (SOO) for a 1D function.

    Parameters:
    f (function): the objective function to optimize
    bounds (tuple): the bounds of the search space (lower, upper)
    num_iterations (int): the number of iterations to run the algorithm
    num_candidates (int): the number of candidate solutions to maintain

    Returns:
    x_best (float): the best solution found
    f_best (float): the value of the objective function at the best solution
    """
    lower, upper = bounds
    x_candidates = np.linspace(lower, upper, num_candidates)
    f_candidates = np.array([f(x) for x in x_candidates])

    for _ in range(num_iterations):
        # Select the most promising candidate
        idx = np.argmin(f_candidates)
        x_best = x_candidates[idx]
        f_best = f_candidates[idx]

        # Generate new candidates by perturbing the current best solution
        new_candidates = np.array([x_best + np.random.uniform(-0.1, 0.1) for _ in range(num_candidates)])
        new_candidates = np.clip(new_candidates, lower, upper)

        # Evaluate the new candidates
        new_f_candidates = np.array([f(x) for x in new_candidates])

        # Update the candidates and their function values
        x_candidates = np.concatenate((x_candidates, new_candidates))
        f_candidates = np.concatenate((f_candidates, new_f_candidates))

        # Remove the worst candidates
        idx = np.argsort(f_candidates)
        x_candidates = x_candidates[idx[:num_candidates]]
        f_candidates = f_candidates[idx[:num_candidates]]

    return x_best, f_best

# Example usage:


bounds = (-10, 10)
x_best, f_best = soo_1d(function, bounds)
print("Best solution:", x_best)
print("Best function value:", f_best)