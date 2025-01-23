"""
Sampling 50 points to compare others experiments
"""
from hpo.core import ModelEval, SearchSpace


def main():
    space = SearchSpace(mode="base",
        savefile="lhs_bis.json")

    evaluator = ModelEval(
        search_space = space,
        dev_run = "",
        experiment_name="lhs_bis",
        model_id="meta-llama/Llama-3.2-1B"
    )

    points = space.LHS(g=50)
    for point in points:
            x = space.get_solution(point)
            y = evaluator.train_and_evaluate(x)
            print(y)