import numpy as np
from copy import deepcopy as dc
import math

class SearchSpace:

    def __init__(self, 
                 mode : str  = "base",
                 savefile : str = None):
        self.variables = {}
        if mode == "base":
            self.base_init()

    def base_init(self):
        space = { 
            "learning_rate" : {"min" : -10,"max" : -1,"type" : "log"},
            "lora_rank" : {"min" : 2,"max" : 32,"type" : "int"},
            "grad_batches" : {"min" : 1,"max" : 16,"type" : "int"},
            "lora_alpha" : {"min" : 16,"max" : 64,"type" : "int"},
            "lora_dropout" : {"min" : 0,"max" : 0.5,"type" : "float"},
            "weight_decay" : {"min" : 0,"max" : 0.5,"type" : "log"} 
        }

        for key, value in space.items():
            variable = value["name"] = key
            self.add_variable(variable)
        

    def get_center(self):
        x = []
        for value in self.variables.values():
            x.append(value.get_center())
        return x
    
    def init_coef(self):
        self.coef = []
        for value in self.variables.values():
            coef = value.init_var_coef()
            self.coef.append(coef)
            
    def add_variable(self, 
                    variable: dict):
        
        new_var = var(
            name = variable["name"],
            vtype = variable["type"],
            min = variable["min"],
            max = variable["max"]
        )
        self.variables[new_var.name] = new_var
    def section(self,K) :
        spaces = [dc(self) for _ in range(K)]
        width = {}
        for var in self.variables.keys():
            width[var] = (self.variables[var]["max"] - self.variables[var]["min"])/self.variables[var]["coef"]



        dim = np.argmax(list(width.values()))
        var = list(self.variables.keys())[dim]

        lower = self.variables[var]["min"]
        upper = self.variables[var]["max"]
        steps = (upper - lower)/K
        for i in range(K) :
            spaces[i].variables[var]["min"] = lower + i*steps
            spaces[i].variables[var]["max"] = lower + (i+1)*steps
            spaces[i].center = spaces[i].get_center()
        return spaces

    def get_solution(self,x):
        sol = Solution(self.variables,x)
        return sol

class Solution(SearchSpace):
    def __init__(self,variables, x):
        self.variables = variables

        self.base_value = x
        self.convert_values(x)
    
    def convert_values(self,x):
        converted_x = [0]*len(x)
        for i,value in enumerate(self.variables.values()):
            converted_x[i] = value.convert_value(x[i])
        self.converted_values = converted_x
    
    def get_values(self):
        return self.converted_values
    
    def speed_run(self):
        res = 1
        for x in self.converted_values:
            res *= x

        return res

    def save(self):
        dic = {}
        dic["base_value"] = self.base_value
        dic["converted_values"] = self.converted_values

class var:

    def __init__(self, name, vtype, min, max):
        self.name = name
        self.type = vtype
        self.min = min
        self.max = max

    def get_center(self): 
        return (self.min + self.max)/2
    
    def convert_value(self,x):
        if self.type == "int":
            return int(x)
        elif self.type == "float":
            return float(x)
        elif self.type == "log":
            return 10**x