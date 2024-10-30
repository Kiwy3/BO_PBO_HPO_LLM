import torch
from pathlib import Path


idx_list = [1,2,3]
name_list = [f"lit_model_{i}.pth" for i in idx_list]
checkpoint_dir = Path("checkpoints/lora")
#key = 'transformer.h.0.attn.attn.weight'
Table_list = []

for name in name_list:
    checkpoint = torch.load(checkpoint_dir / name, weights_only=True)
    Table_list.append(checkpoint)

""" shape = Table_list[0]['transformer.h.0.attn.attn.weight'].shape
n, p = shape """
for key in Table_list[0].keys():
    print("\n",key)
    A = Table_list[0][key]
    B = Table_list[2][key]
    C = A - B 
    if len(C.shape)==2 :
        print("dim_2")
        if all(C[i,j]==0 for i,j in range(A.shape[0],A.shape[1])):
            print("same for ", key)
        else:
            print("different for ", key)
    else :
        print("dim_1")
        if all(C[i]==0 for i in range(A.shape[0])):
            print("same for ", key)
        else:
            print("different for ", key)
