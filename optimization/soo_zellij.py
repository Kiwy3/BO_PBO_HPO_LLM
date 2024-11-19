
from zellij.core.objective import Minimizer
from zellij.utils.benchmarks import himmelblau
from zellij.core import ArrayVar, FloatVar, Loss, Experiment, Threshold, BooleanStop
from zellij.strategies.fractals import DBA, CenterSOO, DBASampling
from zellij.strategies.tools import Section, Min
from zellij.strategies.tools.tree_search import SooTreeSearch
from zellij.utils.converters import FloatMinMax, ArrayDefaultC

lf = Loss( objective=[Minimizer("obj")])(himmelblau)
values = ArrayVar(
    FloatVar("float_1", -5 , 5, converter=FloatMinMax()),
    FloatVar("float_2", -5, 5, converter=FloatMinMax()),
    converter=ArrayDefaultC()
)
sp = Section(values, section=3)

explor = CenterSOO(sp)
stop1 = BooleanStop(explor, "computed")  # set target to None, DBA will automatically asign it.
stop2 = Threshold(lf, "calls", 9)
dba = DBASampling(
    sp, SooTreeSearch(sp, float("inf")), explor, scoring=Min()
)
exp = Experiment(dba, lf, stop2, 
                 save="soo"
                 )
exp.run()
print(f"Best solution:f({lf.best_point})={lf.best_score}")



