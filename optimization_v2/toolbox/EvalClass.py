import os 
import json
import math
import torch
from typing import Dict, List, Literal, Optional, Tuple, Union
from SearchSpace import SearchSpace, Solution, var

experiments = {
    "experiment1": {
        "name" : "experiment1",
        "lora_r" : 8,
        "lora_alpha" : 16,
        "dropout" : 0.05,
        "min_lr" : 1e-5
    },
    "experiment2": {
        "name" : "experiment2",
        "lora_r" : 4,
        "lora_alpha" : 10,
        "dropout" : 0.1,
        "min_lr" : 1e-4
    }
}

class ModelEval:
    def __init__(self,
                search_space : SearchSpace = SearchSpace("base"), 
                model_id : str ="meta-llama/Llama-3.2-1B",
                experiment_name : str = "experiment1",
                dev_run : str = "fake"):
        self.model_id = model_id
        self.task = "hellaswag"
        self.space = search_space
        self.folder = experiment_name
        self.dev_run = dev_run

    def train_and_evaluate(self,
                           x : Solution):
        if self.dev_run == "fake": # fake run for testing
            return x.speed_run()
        
        lora_r, lora_alpha, dropout, min_lr = x.get_values()

        training_string = (f"litgpt finetune "+ #command
                           f"{self.model_id} --out_dir {self.folder}"+ #path management
                           f" --devices {torch.cuda.device_count()}   --precision bf16-true "+ #global parameter of the training
                           f"--train.epochs 1 --train.lr_warmup_steps 100 "+ #Training args
                           f"--eval.interval 150 "+#Eval args
                           f"--lora_key true --lora_value true "+#lora application
                           f"--lora_r {lora_r} --lora_alpha {lora_alpha} --lora_dropout {dropout} --train.min_lr {min_lr}" #hyperparameter
                           )
        eval_string = (f"litgpt evaluate "+ # command
                       f"{self.folder}/final --out_dir eval "+ # path management
                       f"--tasks {self.task} --limit 50 " #tasks definition
                       )
        os.system(training_string)
        os.system(eval_string)

        with open("eval/results.json", "r") as f:
            results = json.load(f)

        cleaning_string = f"rm -rf {self.folder} eval"
        os.system(cleaning_string)
        os.system("rm -rf eval")

        return results["results"][self.task]["acc,none"]


evaluator = ModelEval(
    lora_r=8,
    lora_alpha=16,
)

for val in experiments.values():
    evaluator.train_and_evaluate(
        exp_name=val["name"],
        x=(val["lora_r"],val["lora_alpha"],val["dropout"],val["min_lr"])
    )