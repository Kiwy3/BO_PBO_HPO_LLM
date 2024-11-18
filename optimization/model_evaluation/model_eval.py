import json
import math
import datetime

from model_evaluation import training
from model_evaluation.utils import add_results
from model_evaluation.eval import task_evaluate
""" from __init__ import training, evaluate
from utils import add_results """

class ModelEvaluator:
    def __init__(self, config=None):
        if config is None:
            self.load_config()
        else:
            self.hyperparameters = config["hyperparameters"]
            self.models = config["models"]
            self.experiment = config["experiment"]
        self.hp_key = list(self.hyperparameters.keys())


    def validate(self):
        lora_path = self.experiment["lora_path"]
        tasks = self.experiment["tasks"]
        eval_limit = self.experiment["eval_limit"]
        results = task_evaluate(lora_path,
                            tasks=tasks[0] if len(tasks) == 1 else tasks,
                            limit=eval_limit,
                            force_conversion=True,
                            out_dir="eval/")
        res = {}
        for task in tasks:
            res[task] =  results[task]["acc,none"]
        return res
    
    def load_config(self):    
        with open("optimization/config.json") as f:
            config = json.load(f)
        self.hyperparameters = config["hyperparameters"]
        self.model = config["model"]
        self.experiment = config["experiment"]

    def convert (self,x,i):
        key = list(self.hp_key)[i]
        type = self.hyperparameters[key]["type"]
        if type == "int":
            return int(x[i])
        elif type == "exp":
            return math.exp(x[i])
        elif type == "float":
            return float(x[i])

    def evaluate(self,x, phase = "optimization"):
        hyperparameters = {}
        for i in range(len(self.hyperparameters.keys())):
            key = self.hp_key[i]
            hyperparameters[key] = self.convert(x,i)

        meta_data = {"date":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     "algorithm" : "BO",
                     "phase" : phase,
        }
        # save hyperparameters
        HP = {"hyperparameters" : hyperparameters,
              "meta_data" : meta_data}
        
        # writing in the file   
        export_file = "optimization/export.json"
        with open(export_file, "a+") as outfile:
            json.dump(HP, outfile)
            outfile.write('\n')

        training()
        result = self.validate()
        add_results(results=result,) 

        return result["mmlu"]

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    print(evaluator.evaluate([0.004086771438464067, 17, 8, 40, 0.25, 0.25]))