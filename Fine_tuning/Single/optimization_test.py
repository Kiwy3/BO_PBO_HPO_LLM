import json
from validate_test import running

lr_list = [000.2,000.1,000.3]
rank =   [2,4,6]

HP = {}
out_list = []

for i in range(3):
    HP["learning_rate"] = lr_list[i]
    HP["lora_rank"] = rank[i]
    print(HP)
    with open("HP_config.json","w") as outfile:
        json.dump(HP, outfile)
    out = running()
    print(out)
    out_list.append(out)
print("-----------------------------------------------\n")
print(out_list)

    