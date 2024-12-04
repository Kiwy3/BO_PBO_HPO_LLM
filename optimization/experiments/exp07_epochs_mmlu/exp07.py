import sys,os
#sys.path.insert(0, os.path.abspath("/home/ndavouse/ft_poc/optimization"))
from model_evaluation.model_eval import ModelEvaluator

if  __name__ == "__main__":
    standard_hp = [
        -3,# learning rate
        2,# lora rank
        8,# gradient batches
        16,# lora alpha
        0.25,# lora dropout
        0.25 # weight decay
    ]
    epoch_list = range(5)

    evaluator = ModelEvaluator()

    config = evaluator.load_config("optimization/experiments/exp07_epochs_mmlu/config.json")
    print("config loaded : ",config)
    for i in epoch_list:
        config["experiment"]["epochs"] = i+1
        evaluator = ModelEvaluator(config=config)
        print(evaluator.experiment)
        res = evaluator.evaluate(standard_hp,config_keys=["epochs"])