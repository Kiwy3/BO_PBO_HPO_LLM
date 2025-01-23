"""
Experiment SOO on LLM fine Tuning

"""
from hpo.core import ModelEval, SearchSpace
from hpo.algorithm.soo import SOO



def main():
    space = SearchSpace(mode="base",
        savefile="soo_bis.json",)

    evaluator = ModelEval(
        search_space = space,
        dev_run = "",
        experiment_name="soo_bis",
        model_id="meta-llama/Llama-3.2-1B"
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