import torch
from pathlib import Path


idx_list = [1,2,3]
name_list = [f"lit_model_{i}.pth" for i in idx_list]
#name_list = ["lit_model.pth", "lit_model_unmerged.pth"]
#name_list = ["lit_model_full.pth"]
name_list = ["lit_model_full_merged.pth", "lit_model_full.pth"]
checkpoint_dir = Path("checkpoints/lora")
pretrained_dir = Path("checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0")
#key = 'transformer.h.0.attn.attn.weight'
Table_list = []


for name in name_list:
    checkpoint = torch.load(checkpoint_dir / name, weights_only=True)
    Table_list.append(checkpoint)

""" A = Table_list[0]["transformer.h.1.attn.attn.lora_B"]
for key in Table_list[0].keys():
    if "lora_B" in key:
        if torch.all(A==0):
            print("all B are zeros") """


""" shape = Table_list[0]['transformer.h.0.attn.attn.weight'].shape
n, p = shape """
for key in Table_list[0].keys():
    print("\n",key)
    A = Table_list[0][key]
    B = Table_list[1][key]
    print(A.shape, B.shape)
    C = A - B
    D = (A - B)==0
    if torch.all(D):
        print("Equal")
    else:
        print("Not equal")