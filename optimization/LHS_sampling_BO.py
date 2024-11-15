import json
import torch
import math
import numpy as np
from scipy.stats.qmc import LatinHypercube

def load_config():    
    with open("optimization/config.json") as f:
        config = json.load(f)
    return config["hyperparameters"], config["models"], config["experiment"]


def v1():
    hp_def, model_dict, experiment = load_config()

    grid_number = 3
    grid = {}
    for key in hp_def.keys():
        temp_list = []
        lower_bound = hp_def[key]["min"]
        upper_bound = hp_def[key]["max"]
        for i in range (grid_number):
            temp_list.append([
                lower_bound + (upper_bound - lower_bound) * i / grid_number,
                lower_bound + (upper_bound - lower_bound) * (i+1) / grid_number
            ]
                )
        grid[key] = temp_list
    torch.manual_seed(1234)
    LHS_table = torch.full((len(hp_def.keys()),grid_number),fill_value=False, dtype=torch.bool,)

    print(LHS_table)
    remaining_grid = grid_number
    for i in range(grid_number):
        print("Iteration number",i+1)


        for k,key in enumerate(hp_def.keys()):
            lower_bound = hp_def[key]["min"]
            upper_bound = hp_def[key]["max"]
            upper_bound -= i*(upper_bound-lower_bound)/grid_number
            print("\t hyperparameter : ",key)
            rand_value = np.random.uniform(lower_bound, upper_bound)
            rand_ind = math.trunc(grid_number*(rand_value-lower_bound)/(upper_bound-lower_bound))
            print("\t random value : ",rand_value)
            print("test : ",rand_ind )
            print("\t looking at correspondance")
            grid[key].pop(rand_ind)
            print(grid[key])
            LHS_table[k][rand_ind] = True
        remaining_grid = remaining_grid-1
        print(LHS_table)



def LHS_sampling(hyperparameters,g=10):
    dim = len(hyperparameters)
    from scipy.stats.qmc import LatinHypercube
    LHS = LatinHypercube(d=dim)
    points = LHS.random(n=g)
    for point,key in enumerate(hyperparameters.keys()):
        lower_bound = hyperparameters[key]["min"]
        upper_bound = hyperparameters[key]["max"]
        points[:,point] = points[:,point]*(upper_bound-lower_bound)+lower_bound
    return points

if __name__ == "__main__":
    hp_def, model_dict, experiment = load_config()
    print(LHS_sampling(hp_def))



