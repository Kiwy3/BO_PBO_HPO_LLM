from zellij.core.geometry import Direct
from zellij.strategies import DBA
from zellij.strategies.tools.tree_search import Potentially_Optimal_Rectangle
from zellij.strategies.tools.direct_utils import Sigma2, SigmaInf
from zellij.utils.converters import IntMinmax
from zellij.core.objective import Maximizer
from zellij.core import ContinuousSearchspace, FloatVar,IntVar, ArrayVar, Loss 
#from zellij.utils.benchmarks import himmelblau
#from model_evaluation import evaluate
import math
import json
from model_evaluation.utils import add_results

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
    hyperparameters = {}
    for i in range(len(hp_def.keys())):
        key = list(hp_def.keys())[i]
        hyperparameters[key] = convert(x,i)

    HP = {"hyperparameters" : hyperparameters,
          "experiment" : experiment}
    
    with open(export_file, "a") as outfile:
        json.dump(HP, outfile)
        outfile.write('\n')
    result = 3*x[0] + 2*x[1]
    add_results(result, export_file)
    
    #HP["mmlu_acc"] = evaluate(HP)

    return 1#HP["mmlu_acc"]



def himmelblau(x):
    print(x)
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


if __name__ == "__main__":
    export_file = "optimization/export.json"
    experiment = {"model_id" : "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                  "nb_device" : 2,
                  "epochs" : 1,
                  "device" : "cuda",
                  "fast_run" : True,
                  }
    experiment["model_name"] = model_dict[experiment["model_id"]]

    loss = Loss(objective=Maximizer)(evaluation_function)

    # define the search space
    values = ArrayVar()
    for i in range(hp_def.keys().__len__()):
        key = list(hp_def.keys())[i]
        values.append(
            FloatVar( key, 
                hp_def[key]["min"],
                hp_def[key]["max"]            
            )
        )
                  

    def Direct_al(
    values,
    loss,
    calls,
    verbose=True,
    level=600,
    error=1e-4,
    maxdiv=3000,
    force_convert=False,
    ):

        sp = Direct(
            values,
            loss,
            calls,
            sigma=Sigma2(len(values)),
        )

        dba = DBA(
            sp,
            calls,
                tree_search=Potentially_Optimal_Rectangle(
                sp, level, error=error, maxdiv=maxdiv
            ),
            verbose=verbose,
        )
        dba.run()

        return sp

    sp = Direct_al(values, loss, 50)
    best = (sp.loss.best_point, sp.loss.best_score)
    print(f"Best solution found:f({best[0]})={best[1]}")
    print("\nsolutions",sp.loss.all_solutions)
    print("\nscores",sp.loss.all_scores)