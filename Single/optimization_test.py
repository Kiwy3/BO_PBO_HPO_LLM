import json
from validate_test import BB_eval

lr_list = [0.001,0.002,0.003]
lr_list = [1e-5, 1e-4, 5e-5]
rank =   [4,2,6]

HP = {}
out_list = []

for i in range(3):
    print(f"iteration number {i+1} :")
    HP["learning_rate"] = lr_list[i]
    HP["lora_rank"] = rank[i]
    print(HP)

    out = BB_eval(HP)
    
    print(out)
    print(dir())
    out_list.append(out)
print("-----------------------------------------------\n")
print(out_list)

    