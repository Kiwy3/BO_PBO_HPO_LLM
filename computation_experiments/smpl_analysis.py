import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


exp_name = "lhs"
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
score["hellaswag_norm"] = score["score"].apply(lambda x : x["hellaswag"]["acc_norm,none"])
score.pop("score")
print(score.describe())

# Extract hp
hp_names = ["lora_rank", "lora_alpha", "lora_dropout", "learning_rate", "weight_decay"]
Solution = data["solution"].apply(lambda x : x["base_value"])
Solution = pd.DataFrame(Solution.to_list(),columns=hp_names)



# Box plot
plt.figure(figsize=(6,6))
sns.boxplot(score, saturation=0.5,width=0.4,
            whis=(0,100),
            medianprops={"color": "black", "linewidth": 2},
            palette="pastel",
            )
plt.title("Box plot of sampling algorithm")
plt.savefig("plots/sampling/box_plot.png")
plt.show()

# Mosaic on variable evolution

""" mosaic_str = (
""
AAADD
AAADD
BBBDD
BBBEE
CCCEE
CCCEE

"")
fig, ax = plt.subplot_mosaic(mosaic_str)
plots_names = ["A","B","C","D","E"]

for i in range(len(hp_names)):
      ax[plots_names[i]].scatter(
            Solution.index,
            Solution[hp_names[i]],
            s=3,
            marker = "x"
      )
      ax[plots_names[i]].set_title(hp_names[i]) """







