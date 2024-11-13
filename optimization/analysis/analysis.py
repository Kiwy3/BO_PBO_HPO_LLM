import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_json("../historic/bo.json",lines=True)
X = pd.json_normalize(data["hyperparameters"])
Y = data["results"].apply(lambda x: x["mmlu"])
print(Y)

hp_names = X.columns.to_list()

# Create a (3,2) grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 9))

# Iterate over the hyperparameter names and plot each scatter plot in a separate subplot
for i, name in enumerate(hp_names):
    row = i // 3
    col = i % 3
    axes[row, col].scatter(X[name], Y)
    #axes[row, col].set_title(name)
    axes[row, col].set_xlabel(name)
    axes[row, col].set_ylabel("MMLU")
plt.show()

# Create a new subplot for the evolution of Y by iterations
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(Y)
ax2.set_title("Evolution of MMLU by Iterations")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("MMLU")

# Show the plots
plt.tight_layout()
plt.savefig("bo.png")
plt.show()