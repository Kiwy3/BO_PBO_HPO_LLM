"""
First experiment using BAMSOO to perform HPO on LLM after v2

"""
from hpo.core import ModelEval, SearchSpace
from hpo.algorithm import BaMSOO


def main():

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