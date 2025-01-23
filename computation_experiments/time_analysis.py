import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def extract_time(filenames):
    datas = []
    for filename in filenames:
        file_name = "results/" + filename + ".json"
    
        with open(file_name, 'r') as f:
                for line in f:
                    if "approximated" in line : 
                        #print(line)
                        pass
                    else : 
                        datas.append(json.loads(line))

    data = pd.DataFrame(datas)
    
    """--------------------- Extract score ---------------------- """
    timing = pd.DataFrame(data["timing"])
    timing["opening_time"] = timing["timing"].apply(lambda x : x["opening_time"])
    timing["end_training_time"] = timing["timing"].apply(lambda x : x["end_training_time"])
    timing["ending_time"] = timing["timing"].apply(lambda x : x["ending_time"])

    timing[['opening_time', 'end_training_time', 'ending_time']] = timing[['opening_time', 'end_training_time', 'ending_time']].apply(pd.to_datetime)
    timing.pop("timing")

    return timing

name_list = ["lhs_bis","soo_bis","bo_bis","bamsoo_bis"]
time_extraction = extract_time(name_list)

df_diff = time_extraction.diff(axis=1)
diff_records = df_diff.describe()
print(diff_records)
diff_records.to_csv("score/time_diff.csv")
