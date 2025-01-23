import torch
import pandas as pd
import json
import datetime
from pathlib import Path
import matplotlib.pyplot as plt

from hpo.core.SearchSpace import SearchSpace, Solution
from typing import Dict, List, Literal, Optional, Tuple, Union

from scipy.stats.qmc import LatinHypercube
import numpy as np


#BoTorch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import LogExpectedImprovement

def fun_error():
    print("no function provided")



class BoGp():
    def __init__(self,
                 space : SearchSpace,
                 maximizer : Optional[bool] = True,
                 filename : str = None,
                 obj_fun = fun_error
                 ):
        if filename is not None :
            self.load_from_file(filename)
        else : 
            self.search_space = space
            self.maximizer = maximizer
            self.n_eval = 0
        self.objective = obj_fun

        lower, upper = self.search_space.get_bounds()
        self.lower_bounds = torch.tensor(lower)
        self.upper_bounds = torch.tensor(upper)
        self.bounds = torch.stack((self.lower_bounds, self.upper_bounds)).to(torch.double)

        self.points = []
        self.scores = []

    def scoring(self, 
                X : Solution,
                info : Optional[Dict] = None) -> Tuple[Solution, float]:
        X.info = info
        Y = self.objective(X)*(-1 if not self.maximizer else 1)
        self.n_eval += 1
        return X, Y

    def add_point(self,
                  X : Solution,
                  Y : float):
        self.points.append(X.base_value)
        self.scores.append([Y])

    def LHS_sampling(self,g=10):
        
        LHS = LatinHypercube(d=self.search_space.get_dimensions())
        LHS_points = LHS.random(n=g)
        converted_point = np.empty_like(LHS_points)
        for j,point in enumerate(LHS_points):
            for i,var in enumerate(self.search_space.variables.values()):
                converted_point[j,i] = var.scale_value(point[i])
        return converted_point.tolist()
    
    def initiate(self, g=10):
        lhs_points = self.LHS_sampling(g=g)
        for point in lhs_points:
            X = self.search_space.get_solution(point)
            X, Y = self.scoring(X,info={"phase":"sampling"})
            self.add_point(X, Y)
    def get_points_tensors(self):
        return torch.tensor(self.points, dtype=torch.double), torch.tensor(self.scores, dtype=torch.double)#.unsqueeze(-1)

    def get_new_point(self):
        train_X, train_Y = self.get_points_tensors()

        gp = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=Normalize(d=self.search_space.get_dimensions()),
        outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        self.logEI = LogExpectedImprovement(model=gp, best_f=max(self.scores),maximize=True)
        candidate, acq_value = optimize_acqf(
            #self.logEI, bounds=self.bounds, q=1, num_restarts=5, raw_samples=10,
            self.logEI, bounds=self.bounds, q=1, num_restarts=5, raw_samples=20,
        )
        candidate_list = [candidate[0][i].item() for i in range(len(candidate[0]))]
        
        return self.search_space.get_solution(candidate_list)
    
    def bestof(self): 
        max_index = np.argmax(self.scores)
        max_score = self.scores[max_index]
        max_point = self.points[max_index]
        print("best point : ",
              "\n \t best score : ", max_score,
              "\n \t center : ", max_point)
        return max_point, max_score
    
    def run(self, 
            budget : int = 10,
            init_budget : int = 5):
        self.initiate(g = init_budget)
        while self.n_eval < budget:
            X = self.get_new_point()
            X, Y = self.scoring(X,info={"phase":"optimization"})
            self.add_point(X, Y)
        return self.bestof()