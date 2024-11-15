import torch
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt

#BoTorch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import LogExpectedImprovement
def load_config():    
    with open("optimization/config.json") as f:
        config = json.load(f)
    return config["hyperparameters"], config["models"], config["experiment"]

class BO_GP():
    def __init__(self,config=None,LHS_g=10):
        if config is None:
            self.hyperparameters, self.model, self.experiment = load_config()
        else:
            self.hyperparameters = config["hyperparameters"]
            self.model = config["model"]
            self.experiment = config["experiment"]

        self.dim = len(self.hyperparameters)
        self.lower_bounds = torch.tensor([self.hyperparameters[key]["min"] for key in self.hyperparameters.keys()])
        self.upper_bounds = torch.tensor([self.hyperparameters[key]["max"] for key in self.hyperparameters.keys()])
        self.bounds = torch.stack((self.lower_bounds, self.upper_bounds))
        if Path(self.experiment["historic_file"]).is_file():
            self.X = self.load_points()
        else:
            print("No historic file, Sampling with LHS")
            self.X = torch.tensor(self.LHS_sampling(g=LHS_g))
            self.Y = []
            for x in self.X:
                self.Y.append([self.evaluate(x)])
            self.Y = torch.tensor(self.Y,dtype=torch.double)
            print("Y = ",self.Y)


    def LHS_sampling(self,g=10):
        from scipy.stats.qmc import LatinHypercube
        LHS = LatinHypercube(d=self.dim)
        points = LHS.random(n=g)
        for point,key in enumerate(self.hyperparameters.keys()):
            lower_bound = self.hyperparameters[key]["min"]
            upper_bound = self.hyperparameters[key]["max"]
            points[:,point] = points[:,point]*(upper_bound-lower_bound)+lower_bound
        return points

    def load_points(self):
        data = pd.read_json(self.experiment["historic_file"],lines=True)
        data = data[data.results.notnull()]
        Y = data.results.apply(lambda x: [x["mmlu"]])
        Y = torch.tensor(Y,dtype=torch.double)
        X = pd.json_normalize(data["hyperparameters"])
        X = torch.tensor(X.values,dtype=torch.double)

    def GP_acq(self):
        gp = SingleTaskGP(
        train_X=self.X,
        train_Y=self.Y,
        input_transform=Normalize(d=self.dim),
        outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        self.logEI = LogExpectedImprovement(model=gp, best_f=self.Y.max(),maximize=True)

    def optGP_sol(self):

        candidate, acq_value = optimize_acqf(
            self.logEI, bounds=self.bounds, q=1, num_restarts=5, raw_samples=20,
        )
        candidate_list = [candidate[0][i].item() for i in range(len(candidate[0]))]
        score = self.evaluate(candidate_list)
        self.X = torch.cat((self.X,candidate))
        self.Y = torch.cat((self.Y, torch.tensor([[score]],dtype=torch.double)))

    def evaluate(self,x):
        return (((x[0]**2+x[1]-11)**2) + (((x[0]+x[1]**2-7)**2)))


    def run(self,n = 10):
        for i in range(n):
            print("iteration ",i+1,":")
            print("\t fitting model")
            self.GP_acq()
            print("\t optimizing and evaluate solution")
            self.optGP_sol()


if __name__ == "__main__":
    config = {    
        "hyperparameters": { 
            "dim1" : {"min" : -6,"max" : 6,"type" : "exp"},
            "dim2" : {"min" : -6,"max" : 6,"type" : "int"}
        },
        "model":{
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0":"tiny-llama-1.1b",
            "meta-llama/Meta-Llama-3.1-8B":"Llama-3.1-8B"
        },
        "experiment": {
            "historic_file": "fake.json",
        }
    }
    g=5
    bo = BO_GP(config,LHS_g=g)
    print(bo.X,"\n", bo.Y)
    plt.scatter(bo.X[:,0],bo.X[:,1], c= "red")
    
    bo.run(n=10)
    plt.scatter(bo.X[g:,0],bo.X[g:,1], c= "blue")
    plt.show()
    