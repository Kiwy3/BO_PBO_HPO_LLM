"""
Test the number of epochs for others experiments

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

def ter():
    space = SearchSpace(mode="base",
        savefile="exp13_ter.json")
    
    evaluator = ModelEval(
        search_space = space,
        dev_run = "",
        experiment_name="exp13_bis",
        model_id="meta-llama/Llama-3.2-3B"
    )
    x = space.get_center()
    evaluator.evaluate(folder="meta-llama/Llama-3.2-3B",
                       x=x)

    points = space.LHS(g=10)
    for i in range(1,5):
        evaluator.epochs = i
        for point in points:
            x = space.get_solution(point)
            x.info["epochs"] = i
            y = evaluator.train_and_evaluate(x)
            print(y)

