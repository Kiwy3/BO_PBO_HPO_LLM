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
    """--------------------- Extract specific SOO info ---------------------- """
    depth = data["info"].apply(lambda x : x["depth"])
    depth.name = "depth"

    return depth.value_counts()






soo_depth = load_transform_file("soo_bis")
bamsoo_depth = load_transform_file("bamsoo_bis")
index = soo_depth.index.to_list()+bamsoo_depth.index.to_list()
counts = pd.concat([bamsoo_depth, soo_depth], axis=0).to_list()
algo = ["soo"]*9 +["bamsoo"]*9

df = pd.DataFrame({"depth":index, "count":counts, "algo":algo})
print(df)

plt.figure(figsize=(6, 6))
#plt.title("Depth distribution")
plt.ylabel("evaluation count")
sns.barplot(x="depth", y="count", hue="algo", data=df, palette="Set1",alpha = 0.8)
plt.savefig(f"plots/bamsoo/depth_compar.png")
plt.show()




