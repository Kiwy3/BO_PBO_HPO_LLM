#basic lib
import torch
from pathlib import Path
import json
import math
import pandas as pd

# Bayesian function
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import LogExpectedImprovement

# custom librairies
from model_evaluation import training, evaluate
from utilities import convert, load_config



#from model_evaluation import evaluate
def evaluation_function(x):
    # convert x into hyperparameters
    hyperparameters = {}
    for i in range(len(hp_def.keys())):
        key = list(hp_def.keys())[i]
        hyperparameters[key] = convert(x,i, hp_def)

    # save hyperparameters
    HP = {"hyperparameters" : hyperparameters}

    # writing in the file   
    with open(export_file, "a+") as outfile:
        json.dump(HP, outfile)
        outfile.write('\n')

    training()
    result = evaluate()

    return result["mmlu"]



def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


if __name__ == "__main__":
    export_file = "optimization/export.json"
    hp_def, model_dict, experiment = load_config() 
    experiment["model_name"] = model_dict[experiment["model_id"]]

    # Initiate BoTorch
    lower_bounds = torch.tensor([hp_def[key]["min"] for key in hp_def.keys()])
    upper_bounds = torch.tensor([hp_def[key]["max"] for key in hp_def.keys()])
    bounds = torch.stack((lower_bounds, upper_bounds)
    )
    
    if Path(experiment["historic_file"]).is_file():
        data = pd.read_json(experiment["historic_file"],lines=True)
        data = data[data.results.notnull()]
        Y = data.results.apply(lambda x: [x["mmlu"]])
        Y = torch.tensor(Y,dtype=torch.double)
        X = pd.json_normalize(data["hyperparameters"])
        X = torch.tensor(X.values,dtype=torch.double)
    else:
        X = [(lower_bounds+upper_bounds)/2].to(torch.double)
        #X = torch.tensor(X,dtype=torch.double)
        Y = torch.tensor([evaluation_function(X)],dtype=torch.double)

    print("model initialized")
    for i in range(experiment.get("calls",10)):
        print("iteration ",i+1,":")
        # Define the model
        print("\t creating new model")
        gp = MixedSingleTaskGP(
        train_X=X,
        train_Y=Y,
        cat_dims=[-1],
        input_transform=Normalize(d=len(hp_def.keys())),
        outcome_transform=Standardize(m=1),
        )

        # Optimize the model
        print("\t optimizing model")
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        logEI = LogExpectedImprovement(model=gp, best_f=Y.max(),maximize=True)
        if i ==0:
            solution = torch.tensor([X[1].tolist()],dtype=torch.double)
            print(solution)
            print(math.exp(logEI(solution)))
        candidate, acq_value = optimize_acqf(
            logEI, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
        )



        # Compute the new evaluation
        print("\t computing new evaluation")
        print("\t\t new_solution : ",candidate)
        candidate_list = [candidate[0][i].item() for i in range(len(candidate[0]))]
        score = evaluation_function(candidate_list)
        X = torch.cat((X,candidate))
        Y = torch.cat((Y,
                       torch.tensor([[score]],dtype=torch.double)))
        print("\t\t new_score : ",score)
    
    print("best solution : ",X[Y.argmax()])
    print("best score : ",Y.max())
        

    





    


