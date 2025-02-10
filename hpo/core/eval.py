import os 
import json
import math
import torch
from typing import Dict, List, Literal, Optional, Tuple, Union
from hpo.core.searchspace import SearchSpace, Solution
from datetime import datetime


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
                search_space : SearchSpace = SearchSpace(mode="base"), 
                model_id : str ="meta-llama/Llama-3.2-1B",
                experiment_name : str = "experiment1",
                dev_run : str = "fake"):
        self.model_id = model_id
        self.tasks : List = ["hellaswag","mmlu"]
        self.space = search_space
        self.folder = experiment_name
        self.dev_run = dev_run
        self.epochs = 1

    def clean_and_add_score(self,
                            results_folder,
                            x : Solution):
        # get results and clean it
        with open(f"{results_folder}/results.json", "r") as f:
            evaluation = json.load(f)
        res = evaluation["results"]
        cleaned_results = {}
        for task in self.tasks:
            cleaned_results[task] = res[task]

        # add score to save
        x.add_score(cleaned_results)
        x.save()
        


    def evaluate(self,
                 folder : str = "meta-llama/Llama-3.2-1B",
                 x : Optional[Solution] = None) -> float:
        
        results_folder = f"eval_{folder}"
        # evaluation string
        tasks_str = "'"
        for task in self.tasks:
            tasks_str += task + ","
        tasks_str = tasks_str[:-1] + "'"
        if folder != self.model_id:
            eval_string = (f"litgpt evaluate "+ # command
                        f"{folder}/final --out_dir eval_{folder} "+ # path management
                        f"--tasks  {tasks_str} " #tasks definition
                        )
        else:
            eval_string = (f"litgpt evaluate "+ # command
                        f"{folder} --out_dir evaluation "+ # path management
                        f"--tasks  {tasks_str} " #tasks definition
                        )
            results_folder = "evaluation"
        os.system(eval_string)

        self.clean_and_add_score(results_folder,x)

        


    def train_and_evaluate(self,
                           x : Solution) -> float:
        
        # fake run for testing
        if self.dev_run == "fake": 
            print("Running fake function")
            return x.speed_run()
        
        # get values from solution
        lora_r, lora_alpha, dropout, min_lr, weight_decay = x.get_values()

        # training string
        optimizer_config = ("'{'class_path': 'torch.optim.AdamW', 'init_args': {"+
                f"'lr': {min_lr}, 'weight_decay': {weight_decay}, 'betas': [{0.9}, {0.999}]"+
                "}} '")
        training_string = (f"litgpt finetune "+ #command
                           f"{self.model_id} --out_dir {self.folder}"+ #path management
                           f" --devices {torch.cuda.device_count()}   --precision bf16-true "+ #global parameter of the training
                           f"--train.epochs {self.epochs} --train.lr_warmup_steps 100 --optimizer {optimizer_config} "+ #Training args
                           f"--eval.interval 1000 "+#Eval args
                           f"--lora_key true --lora_value true --lora_query true --lora_head true "+#lora application
                           f"--lora_r {lora_r} --lora_alpha {lora_alpha} --lora_dropout {dropout} " #hyperparameter
                           )
        
        # evaluation string
        tasks_str = "'"
        for task in self.tasks:
            tasks_str += task + ","
        tasks_str = tasks_str[:-1] + "'"
        eval_string = (f"litgpt evaluate "+ # command
                       f"{self.folder}/final --out_dir eval_{self.folder} "+ # path management
                       f"--tasks  {tasks_str} " #tasks definition
                       )
        
        # run and timestamp
        x.opening_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        os.system(training_string)
        x.end_training_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        os.system(eval_string)
        x.ending_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # get results and clean it
        with open(f"eval_{self.folder}/results.json", "r") as f:
            evaluation = json.load(f)
        res = evaluation["results"]
        cleaned_results = {}
        for task in self.tasks:
            cleaned_results[task] = res[task]

        # add score to save
        x.add_score(cleaned_results)
        x.save()

        # cleaning
        cleaning_string = f"rm -rf {self.folder} eval_{self.folder}"
        os.system(cleaning_string)
        os.system("rm -rf eval")

        # return acc (normalized or not) for hpo
        loop_results = cleaned_results[self.tasks[0]]

        return loop_results["acc,none"]