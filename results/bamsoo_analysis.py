import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




file_name = "exp11_bis" + ".json"
datas = []
with open(file_name, 'r') as f:
        for line in f:
            datas.append(json.loads(line))

data = pd.DataFrame(datas)

state = data["info"].apply(lambda x : x["score_state"])

score = data["score"]
approximated = score[state == "approximated"]
UCB = approximated.apply(lambda x : x["UCB"])
LCB = approximated.apply(lambda x : x["LCB"])

evaluated = score[state == "evaluated"]
Y_eval = evaluated.apply(lambda x : x["hellaswag"]["acc_norm,none"])



fig = plt.figure(figsize=(12, 8))
sns.scatterplot(x=evaluated.index, y=Y_eval, label = "evaluated")
sns.scatterplot(x=approximated.index, y=UCB, label = "UCB")
sns.scatterplot(x=approximated.index, y=LCB, label = "LCB")

