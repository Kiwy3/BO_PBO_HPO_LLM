"""
First experiment using SOO to perform HPO on LLM after v2



"""
from optimization_v2.toolbox import ModelEval, SearchSpace
from optimization_v2.algorithm.soo import SOO


def main():
    space = SearchSpace(mode="base",
        savefile="exp10.json",)

    evaluator = ModelEval(
        search_space = space,
        dev_run = "",
        experiment_name="exp10"
    )

    soo = SOO(
        space=space,
        maximizer=True,
        K=3,
        obj_fun=evaluator.train_and_evaluate
    )

    soo.run(
        budget=50,
        saving=True
    )

    soo.print()

def bis():
    space = SearchSpace(mode="base",
        savefile="exp10_bis.json",)

    evaluator = ModelEval(
        search_space = space,
        dev_run = "",
        experiment_name="exp10_bis",
        model_id="meta-llama/Llama-3.2-3B"
    )

    soo = SOO(
        space=space,
        maximizer=True,
        K=3,
        obj_fun=evaluator.train_and_evaluate
    )

    soo.run(
        budget=50,
        saving=True
    )

    soo.print()