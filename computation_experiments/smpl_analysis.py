import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

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
plt.figure(figsize=(10,6))
sns.boxplot(score, saturation=0.5,width=0.4,
            #whis=(0,100),
            medianprops={"color": "black", "linewidth": 2},
            palette="pastel",
            )
plt.title("Box plot of sampling algorithm")
plt.savefig("plots/sampling/lhs_box_plot.png")
plt.show()


print(score.corr())
print(Solution.apply(lambda x : x.corr(score["hellaswag"])))


def days_between(d1, d2):
    d1 = datetime.strptime(d1, '%Y-%m-%d %H:%M:%S')
    d2 = datetime.strptime(d2, '%Y-%m-%d %H:%M:%S')
    sec = abs((d2 - d1).seconds)
    hours = int(sec/3600)
    minute = int((sec - hours*3600)/60)
    return f"{int(sec / 3600)} hours, {minute} minutes and {sec % 60} seconds"
start_date = min(data["timing"].apply(lambda x : x["opening_time"]))
end_date = max(data["timing"].apply(lambda x : x["starting_time"]))

print(days_between(start_date, end_date))







