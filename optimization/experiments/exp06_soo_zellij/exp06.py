# Import Zellij functions
from zellij.core.objective import Minimizer, Maximizer
from zellij.utils.benchmarks import himmelblau
from zellij.core import ArrayVar, FloatVar, Loss, Experiment, Threshold, BooleanStop
from zellij.strategies.fractals import CenterSOO, DBASampling
from zellij.strategies.tools import Section, Min
from zellij.strategies.tools.tree_search import SooTreeSearch
from zellij.utils.converters import FloatMinMax, ArrayDefaultC

# Import custom eval function
from model_evaluation.model_eval import ModelEvaluator


def main():
    evaluator = ModelEvaluator(config_file="optimization/experiments/exp06_soo_zellij/config.json")
    calls = evaluator.experiment["calls"]
    lf = Loss( objective=[Maximizer("obj")])(evaluator)

    # define the search spac
    hp_def = evaluator.hyperparameters
    values = ArrayVar(converter=ArrayDefaultC())
    for i in range(hp_def.keys().__len__()):
        key = list(hp_def.keys())[i]
        values.append(
            FloatVar( key, 
                hp_def[key]["min"],
                hp_def[key]["max"],
                converter=FloatMinMax()            
            )
        )

    sp = Section(values, section=3)

    explor = CenterSOO(sp)
    #stop1 = BooleanStop(explor, "computed")  # set target to None, DBA will automatically asign it.
    stop = Threshold(lf, "calls", calls)
    dba = DBASampling(
        sp, SooTreeSearch(sp, float("inf")), explor, scoring=Min()
    )
    exp = Experiment(dba, lf, stop, 
                    save="soo"
                    )
    exp.run()
    return lf


if __name__ == "__main__":
    lf = main()
    print(f"Best solution:f({lf.best_point})={lf.best_score}")