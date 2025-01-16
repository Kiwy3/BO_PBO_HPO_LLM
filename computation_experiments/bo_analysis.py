import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

exp_name = "bo"
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

plt.figure(figsize=(10, 6))
sns.scatterplot(score)
plt.title("BO : score over iterations")
plt.savefig(f"plots/bo_score_over_time.png")