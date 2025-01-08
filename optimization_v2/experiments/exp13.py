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
        y = evaluator.train_and_evaluate(space.get_center())
        print(y)

def bis():
    space = SearchSpace(mode="base",
        savefile="exp13_bis.json")
    
    evaluator = ModelEval(
        search_space = space,
        dev_run = "",
        experiment_name="exp13_bis",
        model_id="meta-llama/Llama-3.2-3B"
    )

    for i in range(1,5):
        evaluator.epochs = i
        y = evaluator.train_and_evaluate(space.get_center())
        print(y)
