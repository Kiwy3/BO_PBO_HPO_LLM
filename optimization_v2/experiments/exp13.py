"""
First experiment using BO-GP to perform HPO on LLM after v2

"""
from optimization_v2.toolbox import ModelEval, SearchSpace
from optimization_v2.algorithm import BoGp


def main():
    space = SearchSpace(mode="base",
        savefile="exp13.json")
    
    evaluator = ModelEval(
        search_space = space,
        dev_run = "",
        experiment_name="exp13"
    )

    for i in range(1,5):
        evaluator.epochs = i
        y = evaluator.train_and_evaluate()
