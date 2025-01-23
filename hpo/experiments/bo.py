"""
First experiment using BO-GP to perform HPO on LLM after v2

"""
from hpo.core import ModelEval, SearchSpace
from hpo.algorithm import BoGp



def main():
    space = SearchSpace(mode="base",
        savefile="bo_bis.json")

    evaluator = ModelEval(
        search_space = space,
        dev_run = "",
        experiment_name="bo_bis",
        model_id="meta-llama/Llama-3.2-1B"
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