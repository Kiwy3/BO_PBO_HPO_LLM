import json
HP = {"learning_rate": 0.001, "lora_rank": 4}
if False : 
    with open("sample.json","w") as outfile:
        json.dump(HP, outfile)

with open("sample.json") as file:
    data = json.load(file)
    print(data)