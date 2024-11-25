import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound

import torch
import math

def function(x):
    print(x)
    x = x[0] if len(x)==2 else x
    return {"obj":np.sin(x)**3+np.sqrt(x+5)}

X = np.linspace(-5,5,100)
Y = function(X)["obj"]
plt.plot(X,Y,c="blue",label="f(x)")

gen = torch.manual_seed(133)
train_X = torch.rand(5,1, generator=gen,dtype=torch.double)*10-5
print(train_X)
Y = torch.sin(train_X) ** 3 + torch.sqrt(train_X + 5)
#Y = Y + 0.1 * torch.randn_like(Y)  # add some noise
plt.scatter(train_X,Y,label =  "tirage aleatoire")



lower_bounds = torch.tensor(X.min())
upper_bounds = torch.tensor(X.max())
bounds = torch.stack((lower_bounds, upper_bounds)).unsqueeze(-1).to(torch.double)


gp = SingleTaskGP(
    train_X=train_X,
    train_Y=Y,
    input_transform=Normalize(d=1),
    outcome_transform=Standardize(m=1),
    )
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

logEI = LogExpectedImprovement(model=gp, best_f=Y.max(),maximize=True)
ucb = UpperConfidenceBound(gp, beta=0.2)

def mean_sigma(X):
    posterior = gp.posterior(X)
    return posterior.mean.detach().numpy(),posterior.stddev.detach().numpy()

x_list = torch.linspace(-5, 5, 500)
y_mean = []
y_lb = []
y_ub = []
beta = ucb._buffers["beta"]
for x in x_list:
    mean, sigma = mean_sigma(torch.tensor([[x]],dtype=torch.double))
    mean = mean.squeeze(-2).squeeze(-1)
    y_mean.append(mean)
    
    std = torch.sqrt(beta * sigma).detach().numpy()
    y_lb.append(sum(mean - std))
    y_ub.append(sum(mean + std))

plt.plot(x_list,y_mean,c="black",label="mean")
plt.fill_between(x_list,y_lb,y_ub,alpha=0.2,label="uncertainty")
plt.legend()
plt.title("Gaussian Process")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.savefig("plots/gaussian_process.jpg")




""" y_ucb = []
y_ei = []
for x in x_list:
    new_x = torch.tensor([[x]],dtype=torch.double)
    y_ei.append(math.exp(
        logEI(new_x)
    )*10)
    y_ucb.append(
        ucb(new_x).detach().numpy()
    )

plt.plot(x_list,y_ucb,c="black",label="ucb")
plt.legend()


for i in range(10):
    candidate, acq_value = optimize_acqf(
        ucb, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
    )
    plt.scatter(candidate,acq_value,c="red")
    break
plt.show() """






