import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_transform_file(filename):
    file_name = "results/" + filename + ".json"
    datas = []
    with open(file_name, 'r') as f:
            for line in f:
                datas.append(json.loads(line))

    data = pd.DataFrame(datas)
    if "bamsoo" in filename:
         state = data["info"].apply(lambda x : x["score_state"])
         data = data[state == "evaluated"]
    data = data[:50]
    """--------------------- Extract score ---------------------- """
    score = pd.DataFrame(data["score"])
    score["mmlu"] = score["score"].apply(lambda x : x["mmlu"]["acc,none"])
    score["hellaswag"] = score["score"].apply(lambda x : x["hellaswag"]["acc,none"])
    #score["hellaswag_norm"] = score["score"].apply(lambda x : x["hellaswag"]["acc_norm,none"])
    score.pop("score")

    return score

soo_score = load_transform_file("soo_bis")
bamsoo_score = load_transform_file("bamsoo_bis")
bo_score = load_transform_file("bo_bis")

style = {"marker":"x",
         "s":10,
         "alpha" : 0.8}
fig, ax = plt.subplots(1,2,figsize=(16,6))
#plt.scatter(range(0,50),soo_score["hellaswag"],label="SOO-Hellaswag")

ax[0].scatter(range(0,50),soo_score["mmlu"],label="SOO-MMLU",**style)
ax[0].scatter(range(0,50),bamsoo_score["mmlu"],label="BaMSOO-MMLU",**style)
ax[0].scatter(range(0,50),bo_score["mmlu"],label="BO-MMLU",**style)
ax[0].set_ylim(0.2,0.4)
ax[0].legend(loc="lower right")
ax[0].set_title("MMLU score")

ax[1].scatter(range(0,50),soo_score["hellaswag"],label="SOO-Hellaswag",**style)
ax[1].scatter(range(0,50),bamsoo_score["hellaswag"],label="BaMSOO-Hellaswag",**style)
ax[1].scatter(range(0,50),bo_score["hellaswag"],label="BO-Hellaswag",**style)
ax[1].set_ylim(0.2,0.5)
ax[1].set_title("Hellaswag score")
ax[1].legend(loc="lower right")
fig.tight_layout()
plt.savefig(f"plots/global/comparison.png")

