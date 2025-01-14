




"""
-------------------- ADD MMLU TO CORRELATION WHEN I HAVE RESULTS WITH MMLU ------------------------

"""
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
names = ["exp10", "exp11", "exp12"]
datas = []
for name in names : 
    file_name = name + ".json"
    
    with open(file_name, 'r') as f:
        for line in f:
            datas.append(json.loads(line))
data = pd.DataFrame(datas)

# Clean and extract hyperparameters
hp_names = ["lora_rank", "lora_alpha", "lora_dropout", "learning_rate", "weight_decay"]
Solution = data["solution"].apply(lambda x : x["base_value"])
Solution = pd.DataFrame(Solution.to_list(),columns=hp_names)


Y = data["score"].apply(lambda x : x["hellaswag"]["acc,none"])


corr = Solution.apply(lambda x : x.corr(Y))
corr = pd.DataFrame(corr,columns=["correlation"])
print(corr)
corr.to_csv("score/correlation.csv", 
            index_label="hyperparameter")


fig = plt.figure(figsize=(10,4))
plt.title("Correlation with hellaswag accuracy")
sns.heatmap(corr.T, 
            annot=True,
            )
plt.savefig("plots/correlation.png")
plt.show()

fig = plt.figure(figsize=(10,4))
sns.barplot(corr.T
            )
plt.show()

