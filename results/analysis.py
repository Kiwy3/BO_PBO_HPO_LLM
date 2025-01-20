import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# Manage file name depending on interactive env or not

exp_name = "exp12"
file_name = exp_name + ".json"

#file_name = "results/" + file_name

# naming needs
hp = ["lora_rank", "lora_alpha", "lora_dropout", "learning_rate", "weight_decay"]
exp_algo = {
    "exp10" : "SOO",
    "exp11" : "BaMSOO",
    "exp11_bis" : "BaMSOO",
    "exp12" : "BO",
}

# load data
with open(file_name, 'r') as f:
    input = [json.loads(line) for line in f]
data = pd.json_normalize(input)
#if len(data) > 50 : data = data.loc[:49]


# Compute time diff
timing = data[["timing.ending_time","timing.opening_time"]]
timing.rename(
    columns={
        "timing.ending_time": "ending_time",
        "timing.opening_time": "opening_time"
    },
    inplace=True
)
timing = timing.apply(pd.to_datetime)
timing["time_diff"] = (timing["ending_time"] - timing["opening_time"]).dt.total_seconds()/60
time_records = timing.time_diff.describe()

#extract score
score = data[["score.hellaswag.acc_norm,none",
               "score.hellaswag.acc_norm_stderr,none",
               "score.hellaswag.acc_stderr,none",
               "score.hellaswag.acc,none"]]
score.rename(
    columns={
        "score.hellaswag.acc_norm,none": "acc_norm",
        "score.hellaswag.acc_norm_stderr,none": "acc_norm_stderr",
        "score.hellaswag.acc_stderr,none": "acc_stderr",
        "score.hellaswag.acc,none": "acc"
    },
    inplace=True
)
score = score.apply(pd.to_numeric)
score_records = score.describe()
score_records.to_csv(f"score/{exp_name}_score.csv")

# extract base values
base_values = data[["solution.base_value"]]
base_values[hp] = pd.DataFrame(base_values["solution.base_value"].tolist(), index= base_values.index)
base_values.rename(
    columns={
        "solution.base_value": "solution"
    },
    inplace=True
)

# extract converted values
conv_values = data[["solution.converted_values"]]
conv_values[hp] = pd.DataFrame(conv_values["solution.converted_values"].tolist(), index= conv_values.index)
conv_values.rename(
    columns={
        "solution.converted_values": "solution"
    },
    inplace=True
)



"""--------------------------Plots ------------------------"""
# Score over time plots
plt.title(f"{exp_algo[exp_name]} : Score over time")
plt.scatter(score.index, score["acc"], marker = "x")
plt.xlabel("iterations")
plt.ylabel("Hellaswag acc_norm")
if exp_name == "exp12" : plt.axvline(10, color='r', linestyle='dashed', linewidth=1, label="sampling")
plt.legend()
plt.savefig(f"plots/{exp_name}_score_over_time.png")
plt.show()


# Variables over time
hp_short = ["rank", "alpha", "dropout", "lr", "decay"]
fig, ax = plt.subplots(len(hp), 1, sharex=True,figsize=(6, 8))
for i in range(len(hp)):
    ax[i].scatter(base_values.index, base_values[hp[i]], marker = "x")
    ax[i].set_ylabel(hp_short[i])
    if exp_name == "exp12" : ax[i].axvline(10, color='r', linestyle='dashed', linewidth=1, label="sampling")
ax[-1].set_xlabel('iterations')
#fig.suptitle(f"{exp_algo[exp_name]} : hyperparameters over iterations")
plt.savefig(f"plots/{exp_name}_variables_over_time.png")
plt.show()

# score over variables
hp_short = ["rank", "alpha", "dropout", "lr", "decay"]
fig, ax = plt.subplots(1,len(hp), sharey=True, figsize=(12,4))
for i in range(len(hp)):
    ax[i].scatter(base_values[hp[i]],score["acc_norm"], marker = "x")
    ax[i].set_xlabel(hp_short[i])
ax[0].set_ylabel('score')
fig.suptitle(f"{exp_algo[exp_name]} : score by hyperparameters value")
plt.savefig(f"plots/{exp_name}_score_by_hp.png")
plt.show()



