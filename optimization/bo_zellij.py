from zellij.core.geometry import Direct, Soo
from zellij.strategies import DBA, Bayesian_optimization
from zellij.strategies.tools.tree_search import Potentially_Optimal_Rectangle, Soo_tree_search
from zellij.strategies.tools.direct_utils import Sigma2, SigmaInf
from zellij.utils.converters import IntMinmax
from zellij.core.objective import Maximizer
from zellij.core import ContinuousSearchspace, FloatVar,IntVar, ArrayVar, Loss 
#from zellij.utils.benchmarks import himmelblau
#from model_evaluation import evaluate
import math
import json

# custom librairies
#import model_evaluation
#from model_evaluation import training, evaluate

hp_def = { 
   "learning_rate" : {"min" : -10,"max" : -1,"type" : "exp"},
   "lora_rank" : {"min" : 2,"max" : 32,"type" : "int"},
   "grad_batches" : {"min" : 0,"max" : 16,"type" : "int"},
   "lora_alpha" : {"min" : 16,"max" : 64,"type" : "int"},
   "lora_dropout" : {"min" : 0,"max" : 0.5,"type" : "float"},
   "weight_decay" : {"min" : 0,"max" : 0.5,"type" : "float"}, 
   }

model_dict = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0":"tiny-llama-1.1b",
    "meta-llama/Meta-Llama-3.1-8B":"Llama-3.1-8B",

}

def convert(x,i, hyperparameters=hp_def):
    key = list(hyperparameters.keys())[i]
    type = hyperparameters[key]["type"]
    if type == "int":
        return int(x[i])
    elif type == "exp":
        return math.exp(x[i])
    elif type == "float":
        return float(x[i])
    values

#from model_evaluation import evaluate
def evaluation_function(x):
    # convert x into hyperparameters
    hyperparameters = {}
    for i in range(len(hp_def.keys())):
        key = list(hp_def.keys())[i]
        hyperparameters[key] = convert(x,i)

    # save hyperparameters
    HP = {"hyperparameters" : hyperparameters,
          "experiment" : experiment}   
    with open(export_file, "a") as outfile:
        json.dump(HP, outfile)
        outfile.write('\n')
    print(model_evaluation.utils.load_hyperparameters())

    training()
    result = evaluate()

    return result["mmlu"]



def himmelblau(x):
    print(x)
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


if __name__ == "__main__":
    export_file = "optimization/export.json"
    experiment = {"model_id" : "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                  "nb_device" : 2,
                  "epochs" : 1,
                  "device" : "cuda",
                  "fast_run" : False,
                  "eval_limit" : 100,
                  "calls":100
                  }
    experiment["model_name"] = model_dict[experiment["model_id"]]

    loss = Loss(objective=Maximizer)(himmelblau)

    # define the search space
    values = ArrayVar(converter=IntMinmax())
    for i in range(hp_def.keys().__len__()):
        key = list(hp_def.keys())[i]
        values.append(
            FloatVar( key, 
                hp_def[key]["min"],
                hp_def[key]["max"]            
            )
        )
                  

    
    sp = ContinuousSearchspace(values,loss)
    bo = Bayesian_optimization(sp,
                               experiment.get("calls",10))

    best, score = bo.run()
    new_best, new_score = bo.run()
    print(f"Best solution found:\nf({best}) = {score}")