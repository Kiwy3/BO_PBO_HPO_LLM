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
        """
        Initialize ModelEvaluator object.

        Parameters
        ----------
        config : dict, optional
            Experiment configuration. If not provided, load from config.json.

        Attributes
        ----------
        hyperparameters : dict
            Hyperparameter definitions.
        models : dict
            Models available for evaluation.
        experiment : dict
            Experiment settings.
        hp_key : list
            Names of hyperparameters.
        """
        if config is None:
            self.load_config()
        else:
            self.hyperparameters = config["hyperparameters"]
            self.models = config["models"]
            self.experiment = config["experiment"]
        self.hp_key = list(self.hyperparameters.keys())


    def validate(self):
        """
        Evaluate the model on the given tasks.

        Returns
        -------
        dict
            Dictionary with task names as keys and accuracy on that task as values.
        """
        lora_path = self.experiment["lora_path"]
        self.tasks = self.experiment["tasks"]
        eval_limit = self.experiment["eval_limit"]
        if eval_limit == 0:
            eval_limit = None
        results = task_evaluate(lora_path,
                            tasks=self.tasks[0] if len(self.tasks) == 1 else self.tasks,
                            limit=eval_limit,
                            force_conversion=True,
                            out_dir="eval/")
        res = {}
        try:
            for task in self.tasks:
                res[task] =  results[task]["acc_norm,none"]
            return res
        except:
            for task in self.tasks:
                res[task] =  results[task]["acc,none"]
            return res
    
    def load_config(self):    
        with open("optimization/config.json") as f:
            config = json.load(f)
        self.hyperparameters = config["hyperparameters"]
        self.model = config["models"]
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

    def evaluate(self,x, phase = "optimization")->float:
        """
        Evaluate the model with the given hyperparameters.

        Args:
            x (list): Hyperparameters to evaluate.
            phase (str, optional): Phase of the evaluation. Defaults to "optimization".

        Returns:
            float: Result of the evaluation.
        """
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

        return result[self.tasks[0]]
    
    def __call__(self, x):
        return self.evaluate(x)

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    print(evaluator.evaluate([0.004086771438464067, 17, 8, 40, 0.25, 0.25]))