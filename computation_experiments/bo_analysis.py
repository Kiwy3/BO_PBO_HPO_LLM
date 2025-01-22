import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

exp_name = "bo_bis"
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
score.describe().to_csv("score/bo_score.csv")
print(score.describe())

# Extract hp
hp_names = ["lora_rank", "lora_alpha", "lora_dropout", "learning_rate", "weight_decay"]
Solution = data["solution"].apply(lambda x : x["base_value"])
Solution = pd.DataFrame(Solution.to_list(),columns=hp_names)


"""--------------------- Score over iterations figure ---------------------- """
plt.figure(figsize=(6, 4))
sns.scatterplot(x=score.index+1,y=score["hellaswag"],label = "evaluation")
sns.scatterplot(x=score[:10].index+1,y=score[:10]["hellaswag"],label = "sampled points")

max_index = np.argmax(score["hellaswag"])
max_score = score["hellaswag"][max_index]
plt.scatter(max_index+1,max_score,color="red",label="best point (Hellaswag)")

plt.xlim(0,51)
plt.ylim(0.2,0.5)
plt.xlabel("iterations")
plt.ylabel("Hellaswag accuracy")
plt.vlines(10,0.2,0.5,colors="red",linestyles="dashed",label="sampling phase")
plt.legend(loc="lower right")
#plt.title("BO : score over iterations")
plt.savefig(f"plots/bo/score_evolution.png")


"""--------------------- Variables over time ---------------------- """
fig, ax = plt.subplots(5,1,sharex=True,figsize=(4, 6))

for i in range(len(hp_names)):
    ax[i].scatter(score.index+1,Solution[hp_names[i]], marker = "x")
    ax[i].scatter(score[:10].index+1,Solution[:10][hp_names[i]], marker = "x")
    ax[i].scatter(max_index+1,Solution[hp_names[i]][max_index],color="red",marker="x",label="best point (Hellaswag)")
    ax[i].set_ylabel(hp_names[i])
ax[-1].set_xlabel('iterations')
fig.tight_layout()
plt.savefig(f"plots/bo/variables_evolution.png")
plt.show()
