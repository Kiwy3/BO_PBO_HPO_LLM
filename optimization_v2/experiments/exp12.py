"""
First experiment using BO-GP to perform HPO on LLM after v2

"""
from optimization_v2.toolbox import ModelEval, SearchSpace
from optimization_v2.algorithm import BoGp


def main():
    space = SearchSpace(mode="base",
        savefile="exp12.json")

    evaluator = ModelEval(
        search_space = space,
        dev_run = "",
        experiment_name="exp12"
    )

    bo = BoGp(
        space=space,
        maximizer=True,
        obj_fun=evaluator.train_and_evaluate
    )

    bo.run(
        budget=50,
        init_budget=10
    )

    bo.print()
