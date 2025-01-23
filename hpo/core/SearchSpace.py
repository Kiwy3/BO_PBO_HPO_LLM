import numpy as np
from copy import deepcopy as dc
import json
from typing import Dict, List, Literal, Optional, Tuple, Union
from datetime import datetime
from scipy.stats.qmc import LatinHypercube

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
        
    def scale_value(self,x):
        return x*self.coef + self.min
        
    def get_dict(self):
        dic = {}
        dic["name"] = self.name
        dic["type"] = self.type
        dic["min"] = self.min
        dic["max"] = self.max
        return dic


class SearchSpace:

    def __init__(self,
                 variables : Optional[dict[str,dict]] =None , 
                 mode : Optional[str]  = None,
                 savefile : Optional[str] = None):
        self.savefile = savefile
        self.variables = {}
        if variables is not None:
            for key, value in variables.items():
                value["name"] = key
                self.add_variable(value)
        else : 
            
            if mode == "base":
                self.base_init()
        #self.center = self.get_center()
        

    def base_init(self) -> None:
        space = {          
            "lora_rank" : {"min" : 2,"max" : 64,"type" : "int"},
            "lora_alpha" : {"min" : 1,"max" : 64,"type" : "int"},
            "lora_dropout" : {"min" : 0,"max" : 0.5,"type" : "float"},
            "learning_rate" : {"min" : -10,"max" : -1,"type" : "log"},
            "weight_decay" : {"min" : -5,"max" : -1,"type" : "log"} 
            #"grad_batches" : {"min" : 1,"max" : 16,"type" : "int"},
        }

        for key, value in space.items():
            value["name"] = key
            
            self.add_variable(value)
        
    def get_dimensions(self) -> int:
        return len(self.variables)

    def get_center(self,
                   type : str = "solution") -> List:
        x = []
        for value in self.variables.values():
            x.append(value.get_center())
        if type == "solution":
            sol = self.get_solution(x)
            return sol
        elif type == "list":
            return x
    
    def init_coef(self) -> None:
        self.coef = []
        for value in self.variables.values():
            coeff = value.coef()
            self.coef.append(coeff)
            
    def add_variable(self, 
                    variable: dict) -> None:
        coef = variable.get("coef",None)
        new_var = var(
            name = variable["name"],
            vtype = variable["type"],
            min = variable["min"],
            max = variable["max"],
            coef = coef
        )
        self.variables[new_var.name] = new_var

    def section(self,
                K : int = 3) -> List:
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

    def LHS(self,
            g : int = 10) -> List:
        LHS = LatinHypercube(d=self.get_dimensions())
        LHS_points = LHS.random(n=g)
        converted_point = np.empty_like(LHS_points)
        for j,point in enumerate(LHS_points):
            for i,var in enumerate(self.variables.values()):
                converted_point[j,i] = var.scale_value(point[i])
        return converted_point.tolist()

    def get_solution(self,
                     x : List) :
        sol = Solution(savefile=self.savefile,
                       variables=self.variables,
                       x=x)
        return sol

    def get_dict(self) -> dict[str,dict]:
        dic = {}
        for key, value in self.variables.items():
            dic[key] = value.get_dict()
        return dic
    
    def get_bounds(self):
        lower = []
        upper = []
        for value in self.variables.values():
            lower.append(value.min)
            upper.append(value.max)
        return lower,upper

class Solution(SearchSpace):
    def __init__(self,
                 variables : dict[str,var],
                 x : List[float],
                 savefile : Optional[str] = None,
                 ):
        self.variables = variables
        self.savefile = savefile
        self.base_value = x
        self.convert_values(x)
        self.info = {}
        
        self.opening_time = ""
        self.end_training_time = ""
        self.ending_time = ""
    
    def convert_values(self,
                       x : List[float]) -> None:
        converted_x = [0]*len(x)
        for i,value in enumerate(self.variables.values()):
            converted_x[i] = value.convert_value(x[i])
        self.converted_values = converted_x
    
    def get_values(self):
        return self.converted_values



    def speed_run(self) -> float:
        res = 1.
        for x in self.converted_values:
            res *= (x +2)**2
            res -= x
        return res

    def add_score(self,
                  score : Union[float,dict[str,float]]):
        if score is float : 
            self.score = {"score" : score}
        else : 
            self.score = score
    def save(self):
        
        if self.savefile is None :
            return
        time_dic = {
            "opening_time" : self.opening_time,
            "end_training_time" : self.end_training_time,
            "ending_time" : self.ending_time
        }

        sol = {}

        sol["base_value"] = self.base_value
        sol["converted_values"] = self.converted_values
        dic = {
            "timing" : time_dic,
            "solution" : sol,
            "score" : self.score,
            "info" : self.info
        }
        with open(self.savefile,"a") as f:
            json.dump(dic,f)
            f.write("\n")


