import torch
import pandas as pd
import json
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
# HPO
from hpo.core.algorithm import algorithm
from hpo.core.searchspace import SearchSpace, Solution
from typing import Dict, List, Literal, Optional, Tuple, Union, Callable

from scipy.stats.qmc import LatinHypercube
import numpy as np


#BoTorch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import LogExpectedImprovement




class BoGp(algorithm):
    def __init__(self,
                 search_space : SearchSpace,
                 objective_function : Callable,
                 maximizer : Optional[bool] = True) :
        super().__init__(
                    search_space = search_space,
                    objective_function = objective_function,
                    maximizer = maximizer
                )

        # Initiate bounds
        lower, upper = self.search_space.get_bounds()
        self.lower_bounds = torch.tensor(lower)
        self.upper_bounds = torch.tensor(upper)
        self.bounds = torch.stack((self.lower_bounds, self.upper_bounds)).to(torch.double)

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
            X, _ = self.scoring(solution=X,info={"phase":"sampling"})

    def get_points_tensors(self):
        points = [x.get_values() for x in self.historic]
        scores = [x.score for x in self.historic]
        return torch.tensor(points, dtype=torch.double), torch.tensor(scores, dtype=torch.double).unsqueeze(-1)

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
        self.logEI = LogExpectedImprovement(model=gp, best_f=max(train_Y.squeeze(1)),maximize=True)
        candidate, acq_value = optimize_acqf(
            self.logEI, bounds=self.bounds, q=1, num_restarts=5, raw_samples=20,
        )
        candidate_list = [candidate[0][i].item() for i in range(len(candidate[0]))]
        
        return self.search_space.get_solution(candidate_list)

    
    def run(self, 
            budget : int = 10,
            init_budget : int = 5):
        self.initiate(g = init_budget)
        while self.n_eval < budget:
            X = self.get_new_point()
            X, Y = self.scoring(X,info={"phase":"optimization"})
        return self.bestof()