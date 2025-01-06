"""
First experiment using BAMSOO to perform HPO on LLM after v2



"""
from optimization_v2.toolbox import ModelEval, SearchSpace
from optimization_v2.algorithm import BaMSOO


def main():
    space = SearchSpace(mode="base",
        savefile="exp11.json")

    evaluator = ModelEval(
        search_space = space,
        dev_run = "",
        experiment_name="exp11"
    )

    soo = BaMSOO(
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
