import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

exp_name = "bamsoo_bis"
file_name = "results/" + exp_name + ".json"
datas = []
with open(file_name, 'r') as f:
        for line in f:
            datas.append(json.loads(line))

data = pd.DataFrame(datas)
state = data["info"].apply(lambda x : x["score_state"])

approx = data[state == "approximated"]
evaluated = data[state == "evaluated"]

# Extract score 
score_approx = pd.DataFrame(approx["score"])
score_approx["LCB"] = score_approx["score"].apply(lambda x : x["LCB"])
score_approx["UCB"] = score_approx["score"].apply(lambda x : x["UCB"])
score_approx.pop("score")


# Extract score 
score_eval = pd.DataFrame(evaluated["score"])
score_eval["mmlu"] = score_eval["score"].apply(lambda x : x["mmlu"]["acc,none"])
score_eval["hellaswag"] = score_eval["score"].apply(lambda x : x["hellaswag"]["acc,none"])
#score["hellaswag_norm"] = score["score"].apply(lambda x : x["hellaswag"]["acc_norm,none"])
score_eval.pop("score")
score_eval.describe().to_csv("score/bamsoo_score.csv")
print(score_eval.describe())

# Extract hp
hp_names = ["lora_rank", "lora_alpha", "lora_dropout", "learning_rate", "weight_decay"]
Solution = data["solution"].apply(lambda x : x["base_value"])
Solution = pd.DataFrame(Solution.to_list(),columns=hp_names)

"""--------------------- Extract specific SOO info ---------------------- """
depth = data["info"].apply(lambda x : x["depth"])
depth.name = "depth"
"""--------------------- Score over iterations figure ---------------------- """
plt.figure(figsize=(10, 6))
plt.ylim(0.1,0.5)
plt.scatter(score_eval.index,score_eval["hellaswag"],c=depth[state=="evaluated"],label="evaluated",vmin=0,marker="x")
plt.scatter(score_approx.index,score_approx["LCB"],marker="v",c=depth[state=="approximated"],vmin=0,label="LCB",alpha=0.6)
plt.scatter(score_approx.index,score_approx["UCB"],marker="^",c=depth[state=="approximated"],vmin=0,label="UCB",alpha=0.6)

plt.legend(loc="lower right")

#plt.title("SOO : score over iterations")
plt.xlabel("iterations")
plt.ylabel("Hellaswag accuracy")
plt.savefig(f"plots/bamsoo/score_evolution.png")