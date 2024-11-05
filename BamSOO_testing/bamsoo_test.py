from zellij.core.variables import ArrayVar, FloatVar
from zellij.strategies.tools import Hypersphere
from zellij.utils import ArrayDefaultC, FloatMinMax
from zellij.core import Experiment, Loss, Minimizer, Calls
from zellij.strategies.fractals import PHS

@Loss(objective=[Minimizer("obj")])
def himmelblau(x):
    res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2 +15
    return {"obj": res}

a = ArrayVar(
    FloatVar("f1", -5, 5, converter=FloatMinMax()),
    FloatVar("i2", -5, 5, converter=FloatMinMax()),
    converter=ArrayDefaultC(),
)
sp = Hypersphere(a)
opt = PHS(sp, inflation=1)
stop = Calls(himmelblau, 18)
exp = Experiment(opt, himmelblau, stop)
exp.run()
print(f"f({himmelblau.best_point})={himmelblau.best_score}")
print(f"Calls: {himmelblau.calls}")
