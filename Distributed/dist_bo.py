""" --------------Import librairies ------------------------------"""
# Standard librairy
import torch
import numpy as np
import matplotlib.pyplot as plt
import math


# Bayesian function
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import LogExpectedImprovement

import torch.distributed as dist
from dist_model import Dist_eval # Custom evaluation of the function

import warnings
warnings.filterwarnings("ignore") 



import torch, math

def bayesian_optimization(n_BO):
    
    HP = {}
    X = torch.tensor([[-1,4]],dtype=torch.double)
    HP["learning_rate"] = math.exp(X[0][0])
    HP["lora_rank"] = round(X[0][1].item())
    val_loss = Dist_eval(HP)
    Y = torch.tensor([[val_loss]],dtype=torch.double)


    """ --------------Bayesian optimisation-------------------"""
    lower_bound = [-10, 2]
    higher_bound = [-1, 32]
    bounds = torch.stack(
        [torch.tensor(lower_bound),
        torch.tensor(higher_bound)]
        ).to(torch.double)
    n_BO = n_BO


    """ --------------Bayesian iterations-------------------"""
    for i in range(n_BO):
        # Define the model
        gp = MixedSingleTaskGP(
        train_X=X,
        train_Y=Y,
        cat_dims=[-1],
        input_transform=Normalize(d=2),
        outcome_transform=Standardize(m=1),
        )

        # Optimize the model
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        logEI = LogExpectedImprovement(model=gp, best_f=Y.max())
        candidate, acq_value = optimize_acqf(
            logEI, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
        )
        print(candidate)
        # Compute the new evaluation
        HP["learning_rate"] = math.exp(candidate[0][0].item())
        HP["lora_rank"] = round(candidate[0][1].item())
        HP["fast_run"] = False
        val_loss = Dist_eval(HP)

        # Append new data
        Y_candidate = torch.tensor([[val_loss]],dtype=torch.double)
        X = torch.cat((X,candidate))
        Y = torch.cat((Y,Y_candidate))
        
    return X, Y
def tensor_csv(data,name):
    """
    Save a tensor as a CSV file.

    Args:
        data (Tensor): The tensor to save.
        name (str): The name of the CSV file to save the tensor as.

    """

    import pandas as pd
    data = data.numpy()
    df = pd.DataFrame(data)
    df.to_csv(f"{name}.csv")

if __name__ == "__main__":
    X, Y = bayesian_optimization(2)
    # Print the current state of X and Y, and save the plot as "Single_BO.png"
    print(X, Y)
    tensor_csv(X,"X")
    tensor_csv(Y,"Y")

