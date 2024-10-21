"""----------------Import librairies ------------------------------"""
#Standard librairy
import torch
import numpy as np
import matplotlib.pyplot as plt
import math

#NN lib
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split 

#Bayesian function
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import LogExpectedImprovement

from bb_llm import BB_eval # Custom evaluation of the function

import warnings
warnings.filterwarnings("ignore") 




HP = {}
X = torch.tensor([[-1]],dtype=torch.double)
HP["learning_rate"] = math.exp(X)
val_loss = BB_eval(HP)
Y = torch.tensor([[val_loss]],dtype=torch.double)


"""-------------Bayesian optimisation-------------------"""
lower_bound,higher_bound  = -10,-1
bounds = torch.stack([torch.tensor([lower_bound]),torch.tensor([higher_bound])]).to(torch.double)
n_BO = 5

#PLOT
plt.scatter(X,Y,c="red",alpha=0.5)
plt.xlim((lower_bound,higher_bound))
plt.title("Iterative bayesian optimisation")


"""-------------Bayesian iterations-------------------"""
for i in range(n_BO):
    #Define the model
    gp = SingleTaskGP(
    train_X=X,
    train_Y=Y,
    input_transform=Normalize(d=1),
    outcome_transform=Standardize(m=1),
    )

    #Optimize the model
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    logEI = LogExpectedImprovement(model=gp, best_f=Y.max())
    candidate, acq_value = optimize_acqf(
        logEI, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
    )

    #Compte new evaluation
    print(candidate)
    HP["learning_rate"] = math.exp(candidate)
    val_loss = BB_eval(HP)

    #Append new data
    Y_candidate = torch.tensor(val_loss).reshape_as(candidate)
    X = torch.cat((X,candidate))
    Y = torch.cat((Y,Y_candidate))

    #Plot the new point
    plt.scatter(candidate,val_loss,c="black",alpha=0.8)
    plt.text(candidate,val_loss+0.02,str(i))

print(X, Y)
plt.savefig("Single_BO.png")

