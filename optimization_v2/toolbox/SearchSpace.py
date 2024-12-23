import numpy as np
from copy import deepcopy as dc
import math
from typing import Dict, List, Literal, Optional, Tuple, Union

class SearchSpace:

    def __init__(self,
                 variables : Optional[dict] =None , 
                 mode : str  = "base",
                 savefile : str = None):
        self.savefile = savefile
        if variables is not None:
            for key, value in variables.items():
                variable = value["name"] = key
                self.add_variable(variable)
        else : 
            self.variables = {}
            if mode == "base":
                self.base_init()
        self.center = self.get_center()
        

    def base_init(self):
        space = {          
            "lora_rank" : {"min" : 2,"max" : 32,"type" : "int"},
            "lora_alpha" : {"min" : 16,"max" : 64,"type" : "int"},
            "lora_dropout" : {"min" : 0,"max" : 0.5,"type" : "float"},
            "learning_rate" : {"min" : -10,"max" : -1,"type" : "log"},
            #"grad_batches" : {"min" : 1,"max" : 16,"type" : "int"},
            #"weight_decay" : {"min" : 0,"max" : 0.5,"type" : "log"} 
        }

        for key, value in space.items():
            value["name"] = key
            self.add_variable(value)
        

    def get_center(self):
        x = []
        for value in self.variables.values():
            x.append(value.get_center())
        return self.get_solution(x)
    
    def init_coef(self):
        self.coef = []
        for value in self.variables.values():
            coef = value.get_coef()
            self.coef.append(coef)
            
    def add_variable(self, 
                    variable: dict):
        
        coef = variable.get("coef",None)
        
        new_var = var(
            name = variable["name"],
            vtype = variable["type"],
            min = variable["min"],
            max = variable["max"],
            coef = coef
        )


        self.variables[new_var.name] = new_var
    def section(self,K) :
        spaces = [dc(self) for _ in range(K)]
        width = {}
        for key, value in self.variables.items():
            width[key] = value.get_norm_width()

        dim_index = np.argmax(list(width.values()))
        key_max = list(self.variables.keys())[dim_index]
        max_var = self.variables[key_max]

        lower = max_var.min
        upper = max_var.max
        steps = (upper - lower)/K
        for i in range(K) :
            spaces[i].variables[key_max].min = lower + i*steps
            spaces[i].variables[key_max].max = lower + (i+1)*steps
        return spaces

    def get_solution(self,x):
        sol = Solution(savefile=self.savefile,
                       variables=self.variables,
                       x=x)
        return sol

    def get_dict(self):
        dic = {}
        for key, value in self.variables.items():
            dic[key] = value.get_dict()
        return dic

class Solution(SearchSpace):
    def __init__(self,savefile,variables, x):
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

    def __init__(self, name : str,
                vtype : str,
                min : float,
                max : float,
                coef : float):
        self.name = name
        self.type = vtype
        self.min = min
        self.max = max

        if coef is None:
            self.init_coef()
        else:
            self.coef = coef

    def get_norm_width(self):
        return (self.max - self.min)/self.coef


    def init_coef(self):
        self.coef = self.max - self.min

    def get_center(self): 
        return (self.min + self.max)/2
    
    def convert_value(self,x):
        if self.type == "int":
            return int(x)
        elif self.type == "float":
            return float(x)
        elif self.type == "log":
            return 10**x
        
    def get_dict(self):
        dic = {}
        dic["name"] = self.name
        dic["type"] = self.type
        dic["min"] = self.min
        dic["max"] = self.max
        return dic