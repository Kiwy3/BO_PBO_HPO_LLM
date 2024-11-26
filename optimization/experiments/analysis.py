import pandas as pd
import matplotlib.pyplot as plt
import json, math

folder_path = "exp03_bo_LHS/"
task = "hellaswag"

with open(f"{folder_path}/config.json") as f:
    config = json.load(f)
experiment = config["experiment"]
hyperparameters = config["hyperparameters"]

data = pd.read_json(folder_path+"export.json",lines=True)
X = pd.json_normalize(data["hyperparameters"])
Y = data["results"].apply(lambda x: x[task])

phase = data["meta_data"].apply(lambda x: x["phase"])
ph_mask = phase == "sampling"
opt, smpl = phase.value_counts()

hp_names = X.columns.to_list()

# Create a (3,2) grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Iterate over the hyperparameter names and plot each scatter plot in a separate subplot
for i, name in enumerate(hp_names):
    row = i // 3
    col = i % 3
    axes[row, col].scatter(X[name][ph_mask], Y[ph_mask], label = "sampling", c="orange")
    axes[row, col].scatter(X[name][~ph_mask], Y[~ph_mask], label = "sampling", c="blue")
    axes[row, col].set_title(name)
    axes[row, col].set_xlabel(name)
    axes[row, col].set_ylabel(task)
    if hyperparameters[name]["type"] != "exp":
        axes[row, col].set_xlim(hyperparameters[name]["min"], hyperparameters[name]["max"])
    else : 
        axes[row, col].set_xlim(math.exp(hyperparameters[name]["min"]),
            math.exp( hyperparameters[name]["max"]))
plt.savefig(folder_path+"score_by_variable.png")
plt.show()

# Create a (3,2) grid of subplots
fig, axes = plt.subplots(3, 2, figsize=(16, 10))

# Iterate over the hyperparameter names and plot each scatter plot in a separate subplot
for i, name in enumerate(hp_names):
    row = i // 2
    col = i % 2
    axes[row, col].scatter(range(len(X[name][ph_mask])), X[name][ph_mask], label = "sampling", c="orange")
    axes[row, col].scatter(range(len(X[name][~ph_mask])), X[name][~ph_mask], label = "sampling", c="blue")
    axes[row, col].set_title(name)
    axes[row, col].set_xlabel(name)
    axes[row, col].set_ylabel(task)
    if hyperparameters[name]["type"] != "exp":
        axes[row, col].set_ylim(hyperparameters[name]["min"], hyperparameters[name]["max"])
    else : 
        axes[row, col].set_ylim(
            math.exp(hyperparameters[name]["min"]),
            math.exp( hyperparameters[name]["max"]))
fig.suptitle("Evolution of "+task+" by Variable")
fig.tight_layout()
plt.savefig(folder_path+"variable_over_time.png")
plt.show()


# Create a new subplot for the evolution of Y by iterations
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(range(len(Y[ph_mask])),(Y[ph_mask]),label = "sampling", c="orange")
ax2.scatter(range(smpl,len(Y)),(Y[~ph_mask]),label = "optimization")
ax2.set_title("Evolution of "+task+" by Iterations")
ax2.set_xlabel("Iteration")
ax2.set_ylabel(task)
ax2.legend()

# Show the plots
plt.tight_layout()
plt.savefig(folder_path+"score_by_iteration.png")
plt.show()

print(Y.max())