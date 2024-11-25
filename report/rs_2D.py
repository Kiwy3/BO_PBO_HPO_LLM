from zellij.utils.benchmarks import himmelblau


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

lower = -5
upper = 5
full_gap = upper - lower

fig, ax = plt.subplots(figsize=(8, 6))
x = y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = himmelblau([X, Y])["obj"]

map = ax.contourf(X, Y, Z, cmap="plasma", levels=100)
fig.colorbar(map)
rand_x = np.random.uniform(lower, upper, 30)
rand_y = np.random.uniform(lower, upper, 30)

plt.scatter(rand_x,rand_y,c="cyan",s=3)    
plt.title("Random Search")
plt.savefig("plots/random_search.jpg")

plt.show()



