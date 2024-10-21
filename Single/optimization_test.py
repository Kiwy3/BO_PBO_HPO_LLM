import json
from validate_test import BB_eval


lr_list = [1e-2, 1e-4, 1e-5]
rank =   [16,8,4]
gb_list = [6,8,16]

HP = {}
out_list = []

for i in range(len(lr_list)):
    print(f"iteration number {i+1} :")
    HP["learning_rate"] = lr_list[i]
    HP["lora_rank"] = rank[i]
    HP["grad_batches"] = gb_list[i]
    print(HP)

    out = BB_eval(HP)
    out_list.append(out)
print("-----------------------------------------------\n\n\n\n\n")
print(out_list)

    