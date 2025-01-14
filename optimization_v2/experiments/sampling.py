"""
Sampling 50 points to compare others experiments
"""
from optimization_v2.toolbox import ModelEval, SearchSpace


def main():
    space = SearchSpace(mode="base",
        savefile="lhs.json")

    evaluator = ModelEval(
        search_space = space,
        dev_run = "",
        experiment_name="lhs",
        model_id="meta-llama/Llama-3.2-3B"
    )

    points = space.LHS(g=50)
    for point in points:
            x = space.get_solution(point)
            y = evaluator.train_and_evaluate(x)
            print(y)