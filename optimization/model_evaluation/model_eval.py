import json
import math
import datetime
import torch
import lightning as L
import litgpt
import os
from pathlib import Path

from model_evaluation import training
from model_evaluation.eval import task_evaluate

from model_evaluation.train_model import LLM_model, merge_lora_weights, lora_filter
from model_evaluation.train_model import LLMDataModule

class ModelEvaluator:
    def __init__(self, config=None, config_file= "optimization/config.json"):
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
            self.__load_config(config_file=config_file)
        else:
            self.hyperparameters = config["hyperparameters"]
            self.models = config["models"]
            self.experiment = config["experiment"]
        self.model_id = self.experiment["model_id"]
        self.model_name = self.models[self.model_id] 
        self.hp_key = list(self.hyperparameters.keys())
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.3"

    def __load_config(self,config_file):    
        """
        Load configuration from a JSON file.

        Parameters
        ----------
        config_file : str
            Path to the configuration file.

        Notes
        -----
        The configuration file should contain three keys: "hyperparameters",
        "models", and "experiment". The value of "hyperparameters" should be a
        dictionary with hyperparameter names as keys and a dictionary with
        "min", "max", and "type" as values. The value of "models" should be a
        dictionary with model names as keys and a string as value. The value
        of "experiment" should be a dictionary with experiment settings as
        values.
        """
        with open(config_file) as f:
            config = json.load(f)
        self.hyperparameters = config["hyperparameters"]
        self.model = config["models"]
        self.experiment = config["experiment"]

    def training(self,x):
        # convert hyperparameters to variables
        learning_rate, lora_rank, grad_batches, lora_alpha, lora_dropout, weight_decay = x

        # data module management
        data_loader = LLMDataModule(
            val_split_fraction=0.05,  # Adjust as needed
            repo_id=self.experiment["dataset"], 
        )
        data_loader.connect(
            tokenizer=litgpt.Tokenizer(f"checkpoints/{self.experiment['model_id']}"),
            batch_size=1,
            max_seq_length=512
        )   

        # create a trainer instance 
        trainer = L.Trainer(
            devices=self.experiment["nb_device"],
            max_epochs=self.experiment["epochs"] if self.experiment["epochs"] > 0 else 1,
            max_steps=20 if self.experiment["fast_run"] else 20000,
            strategy=self.experiment["strategy"],
            accumulate_grad_batches=grad_batches,
            precision="16-mixed",
            enable_checkpointing=True,
            #plugins=quantize_plug(),
        )
    

        # model configuration
        model = LLM_model(
            low_rank=lora_rank, 
            rate=learning_rate,
            l_alpha=lora_alpha,
            l_dropout=lora_dropout,
            weight_decay = weight_decay,
            model_name=self.model_name,
            model_id=self.model_id        
        ).to(self.experiment["device"])
        trainer.fit(model, datamodule=data_loader)

        # Saving merged model
        print("\t merging and saving")
        merge_lora_weights(model.model)
        state_dict = {k.replace("linear.", ""): v for k, v in model.model.state_dict().items() if not lora_filter(k, v)}
        save_path = Path("checkpoints/lora") / "lit_model.pth"
        torch.save(state_dict, save_path)


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
    


    def __add_results(self, results):

        with open(self.experiment["historic_file"], 'r+') as f:
            lines = f.readlines()
            last_line = lines[-1]
            last_line_data = json.loads(last_line)
            last_line_data["meta_data"]["end_date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            last_line_data['results'] = results
            lines[-1] = json.dumps(last_line_data) + '\n'
            f.seek(0)
            f.writelines(lines)
            f.truncate()

    def __variable_conversion (self,x,i):
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
        #torch.set_float32_matmul_precision('medium' | 'high')
        hyperparameters = {}
        for i in range(len(self.hyperparameters.keys())):
            key = self.hp_key[i]
            hyperparameters[key] = self.__variable_conversion(x,i)

        meta_data = {"start_date":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     "algorithm" : "BO",
                     "phase" : phase,
        }
        # save hyperparameters
        HP = {"hyperparameters" : hyperparameters,
              "meta_data" : meta_data}
        
        # writing in the file   
        export_file = self.experiment["historic_file"]
        with open(export_file, "a+") as outfile:
            json.dump(HP, outfile)
            outfile.write('\n')

        training()
        result = self.validate()
        self.__add_results(results=result,) 

        return result[self.tasks[0]]
    
    def __call__(self, x):
        return self.evaluate(x)

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    print(evaluator.evaluate([0.004086771438464067, 17, 8, 40, 0.25, 0.25]))