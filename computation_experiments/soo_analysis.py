import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

exp_name = "soo_bis"
file_name = "results/" + exp_name + ".json"
datas = []
with open(file_name, 'r') as f:
        for line in f:
            datas.append(json.loads(line))

data = pd.DataFrame(datas)


"""--------------------- Extract score ---------------------- """

score = pd.DataFrame(data["score"])
score["mmlu"] = score["score"].apply(lambda x : x["mmlu"]["acc,none"])
score["hellaswag"] = score["score"].apply(lambda x : x["hellaswag"]["acc,none"])
#score["hellaswag_norm"] = score["score"].apply(lambda x : x["hellaswag"]["acc_norm,none"])
score.pop("score")
score.describe().to_csv("score/soo_score.csv")
print(score.describe())

"""--------------------- Extract variables ---------------------- """
hp_names = ["lora_rank", "lora_alpha", "lora_dropout", "learning_rate", "weight_decay"]
Solution = data["solution"].apply(lambda x : x["base_value"])
Solution = pd.DataFrame(Solution.to_list(),columns=hp_names)

"""--------------------- Extract specific SOO info ---------------------- """
depth = data["info"].apply(lambda x : x["depth"])
depth.name = "depth"
"""--------------------- Score over iterations figure ---------------------- """
plt.figure(figsize=(10, 6))
plt.ylim(0.2,0.5)
sns.scatterplot(x=score.index+1,y=score["hellaswag"], hue = depth, palette="deep")
#plt.title("SOO : score over iterations")
plt.xlabel("iterations")
plt.ylabel("Hellaswag accuracy")
plt.savefig(f"plots/soo/score_evolution.png")


"""--------------------- Score over iterations figure ---------------------- """
fig, ax = plt.subplots(5,1,sharex=True,figsize=(4, 6))

for i in range(len(hp_names)):
    sns.scatterplot(ax=ax[i],x=score.index+1,y=Solution[hp_names[i]], hue = depth, palette="deep",legend=False)
    ax[i].set_ylabel(hp_names[i])
ax[-1].set_xlabel('iterations')
fig.tight_layout()
plt.savefig(f"plots/soo/variables_evolution.png")
plt.show()