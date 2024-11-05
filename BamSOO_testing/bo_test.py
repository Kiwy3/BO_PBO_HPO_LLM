from zellij.core import UnitSearchspace, ArrayVar, FloatVar
from zellij.utils import FloatMinMax, ArrayDefaultC
from zellij.core import Experiment, Loss, Minimizer, Calls
from zellij.strategies.continuous import BayesianOptimization

# Additional imports for BamSoo
from zellij.strategies.fractals.bo.bamsoo import BaMSOO
from zellij.strategies.fractals import  CenterSOO
from zellij.strategies.tools import Section


@Loss(objective=[Minimizer("obj")])
def himmelblau(x):
    res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2 + (x[2] + x[0] ** 2 - 7) ** 2
    return {"obj": res}


a = ArrayVar(
    FloatVar("f1", -5, 5, converter=FloatMinMax()),
    FloatVar("i2", -5, 5, converter=FloatMinMax()),
    FloatVar("i3", -5, 5, converter=FloatMinMax()),
    converter= ArrayDefaultC(),
)
#sp = UnitSearchspace(a)
#opt = BayesianOptimization(sp)
sp = Section(a, section=3)
explor = CenterSOO(sp)
opt = BaMSOO(sp,explor,nu = 0.5)
stop = Calls(himmelblau, 10)
exp = Experiment(opt, himmelblau, stop)
exp.run()
print(f"f({himmelblau.best_point})={himmelblau.best_score}")
print(f"Calls: {himmelblau.calls}")
