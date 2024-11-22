import torch
import pandas as pd
import json
import datetime
from pathlib import Path
import matplotlib.pyplot as plt

from model_evaluation import ModelEvaluator

#BoTorch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import LogExpectedImprovement

class BO_HPO(ModelEvaluator):
    def __init__(self,config=None):
        ModelEvaluator.__init__(self, config=config)
        self.dim = len(self.hyperparameters)
        self.lower_bounds = torch.tensor([self.hyperparameters[key]["min"] for key in self.hyperparameters.keys()])
        self.upper_bounds = torch.tensor([self.hyperparameters[key]["max"] for key in self.hyperparameters.keys()])
        self.bounds = torch.stack((self.lower_bounds, self.upper_bounds)).to(torch.double)

    def init(self, LHS_g=10):
        if Path(self.experiment["historic_file"]).is_file():
            self.X = self.load_points()
        else:
            print("No historic file, Sampling with LHS")
            #create file
            f = open(self.experiment["historic_file"], "w")
            f.close()
            #Define point to evaluate
            self.X = torch.tensor(self.LHS_sampling(g=LHS_g))
            self.Y = []
            for x in self.X:
                self.Y.append([self.evaluate(x,phase="sampling")])
            self.Y = torch.tensor(self.Y,dtype=torch.double)

    def LHS_sampling(self,g=10):
        from scipy.stats.qmc import LatinHypercube
        LHS = LatinHypercube(d=self.dim)
        points = LHS.random(n=g)
        for point,key in enumerate(self.hyperparameters.keys()):
            lower_bound = self.hyperparameters[key]["min"]
            upper_bound = self.hyperparameters[key]["max"]
            points[:,point] = points[:,point]*(upper_bound-lower_bound)+lower_bound
        return points

    def load_points(self): #When there is a historic file
        data = pd.read_json(self.experiment["historic_file"],lines=True)
        data = data[data.results.notnull()]
        Y = data.results.apply(lambda x: [x[self.experiment["tasks"][0]]])
        Y = torch.tensor(Y,dtype=torch.double)
        X = pd.json_normalize(data["hyperparameters"])
        X = torch.tensor(X.values,dtype=torch.double)

    def new_point(self):
        gp = SingleTaskGP(
        train_X=self.X,
        train_Y=self.Y,
        input_transform=Normalize(d=self.dim),
        outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        self.logEI = LogExpectedImprovement(model=gp, best_f=self.Y.max(),maximize=True)
        candidate, acq_value = optimize_acqf(
            #self.logEI, bounds=self.bounds, q=1, num_restarts=5, raw_samples=10,
            self.logEI, bounds=self.bounds, q=1, num_restarts=5, raw_samples=20,
        )
        candidate_list = [candidate[0][i].item() for i in range(len(candidate[0]))]
        score = self.evaluate(candidate_list)
        self.X = torch.cat((self.X,candidate))
        self.Y = torch.cat((self.Y, torch.tensor([[score]],dtype=torch.double)))


    def eval_benchmark(self,x,name="Rosenbrock"):
        if name == "Rosenbrock":
            y = 0
            for i in range(len(x)-1):
                y += 100*( (x[i+1]**2-x[i]**2)**2 + ((x[i+1]-1)**2) )
            return y
        elif name == "Himmelblau":
            return (((x[0]**2+x[1]-11)**2) + (((x[0]+x[1]**2-7)**2)))


    def run(self,n = 10):
        for i in range(n):
            print("iteration ",i+1,":")
            self.new_point()
            print("\t new point : ",self.X[-1], "\n\t new score : ",self.Y[-1])



if __name__ == "__main__":

    g=10
    bo = BO_HPO(LHS_g=g)
    print(bo.X,"\n", bo.Y, bo.bounds)
    bo.run(n=50)
    