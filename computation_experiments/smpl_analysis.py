import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

exp_name = "lhs_bis"
file_name = "results/" + exp_name + ".json"
datas = []
with open(file_name, 'r') as f:
        for line in f:
            datas.append(json.loads(line))

data = pd.DataFrame(datas)


# Extract score 
score = pd.DataFrame(data["score"])
score["mmlu"] = score["score"].apply(lambda x : x["mmlu"]["acc,none"])
score["hellaswag"] = score["score"].apply(lambda x : x["hellaswag"]["acc,none"])
#score["hellaswag_norm"] = score["score"].apply(lambda x : x["hellaswag"]["acc_norm,none"])
score.pop("score")
score.describe().to_csv(f"score/{exp_name}_score.csv")
print(score.describe())

# Extract hp
hp_names = ["lora_rank", "lora_alpha", "lora_dropout", "learning_rate", "weight_decay"]
Solution = data["solution"].apply(lambda x : x["base_value"])
Solution = pd.DataFrame(Solution.to_list(),columns=hp_names)

def days_between(d1, d2):
    d1 = datetime.strptime(d1, '%Y-%m-%d %H:%M:%S')
    d2 = datetime.strptime(d2, '%Y-%m-%d %H:%M:%S')
    sec = abs((d2 - d1).seconds)
    hours = int(sec/3600)
    minute = int((sec - hours*3600)/60)
    return f"{int(sec / 3600)} hours, {minute} minutes and {sec % 60} seconds"
start_date = min(data["timing"].apply(lambda x : x["opening_time"]))
end_date = max(data["timing"].apply(lambda x : x["ending_time"]))

print(days_between(start_date, end_date))
print(score.corr())
print(Solution.apply(lambda x : x.corr(score["hellaswag"])))

# Box plot

concatenated = pd.concat((score,Solution),axis=1)
correlation = concatenated.corr()

# box plot figure
plt.figure(figsize=(6,6))
sns.boxplot(score, saturation=0.5,width=0.4,
            #whis=(0,100),
            medianprops={"color": "black", "linewidth": 2},
            palette="pastel",
            )
plt.title("Box plot of sampling algorithm")
plt.savefig("plots/sampling/lhs_box_plot.png")
plt.show()

# heatmap figure
plt.figure(figsize=(4,6))
sns.heatmap(correlation[["mmlu","hellaswag"]],
    cmap = sns.diverging_palette(230, 20, as_cmap=True),
    vmin=-1,
    annot=True,
    cbar=False
    )
plt.title("Correlation heatmap")
plt.tight_layout()
plt.savefig("plots/sampling/lhs_correlation.png")
plt.show()


# concatenated subplots
fig, ax = plt.subplots(1,2,
                       figsize=(10,5),
                       width_ratios=[1, 2])
sns.heatmap(ax=ax[0],
    data=correlation[["mmlu","hellaswag"]],
    cmap = sns.diverging_palette(230, 20, as_cmap=True),
    vmin=-1,
    annot=True,
    cbar=False
    )
ax[0].set_title("Correlation heatmap")

sns.boxplot(ax=ax[1],
            data=score, saturation=0.5,width=0.4,
            #whis=(0,100),
            medianprops={"color": "black", "linewidth": 2},
            palette="pastel",
            )
ax[1].set_title("Box plot of sampling algorithm")

fig.suptitle("Sampling experiment",
             size=16)

plt.savefig("plots/sampling/lhs.png")
plt.show()
