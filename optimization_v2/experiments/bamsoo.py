"""
First experiment using BAMSOO to perform HPO on LLM after v2

"""
from optimization_v2.toolbox import ModelEval, SearchSpace
from optimization_v2.algorithm import BaMSOO


def main():
    space = SearchSpace(mode="base",
        savefile="bamsoo.json")

    evaluator = ModelEval(
        search_space = space,
        dev_run = "",
        experiment_name="bamsoo",
        model_id="meta-llama/Llama-3.2-3B"
    )

    bamsoo = BaMSOO(
        space=space,
        maximizer=True,
        K=3,
        obj_fun=evaluator.train_and_evaluate,
        eta=0.8
    )

    bamsoo.run(
        budget=50,
        saving=True
    )

    bamsoo.print()

def bis():
    space = SearchSpace(mode="base",
        savefile="bamsoo_bis.json")

    evaluator = ModelEval(
        search_space = space,
        dev_run = "",
        experiment_name="bamsoo_bis",
        model_id="meta-llama/Llama-3.2-1B"
    )

    bamsoo = BaMSOO(
        space=space,
        maximizer=True,
        K=3,
        obj_fun=evaluator.train_and_evaluate,
        eta=1
    )

    bamsoo.run(
        budget=50,
        saving=True
    )

    bamsoo.print()