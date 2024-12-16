from algorithm import SOO, array
from model_evaluation.model_eval import ModelEvaluator

if  __name__ == "__main__":
    objective = ModelEvaluator(config_file="optimization/experiments/exp08_soo_manual/config.json")
    hyperparameters = objective.hyperparameters
    space = array(hyperparameters)

    so = SOO(filename="tree.json",
             obj_fun=objective)
    so.run(
        budget=20,
        saving=True
    )
