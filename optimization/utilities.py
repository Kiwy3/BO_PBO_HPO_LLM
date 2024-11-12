import math
import json

def convert(x,i, hyperparameters):
    key = list(hyperparameters.keys())[i]
    type = hyperparameters[key]["type"]
    if type == "int":
        return int(x[i])
    elif type == "exp":
        return math.exp(x[i])
    elif type == "float":
        return float(x[i])
    
def load_config():
    with open("optimization/config.json") as f:
        config = json.load(f)
    return config["hyperparameters"], config["models"], config["experiment"]