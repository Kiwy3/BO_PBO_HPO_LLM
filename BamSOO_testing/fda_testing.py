from zellij.utils.benchmarks import himmelblau
from zellij.core import ArrayVar, FloatVar, Loss, Experiment, Threshold
from zellij.core.objective import Minimizer
from zellij.strategies.fractals import PHS, ILS, DBA
from zellij.strategies.tools import Hypersphere
from zellij.strategies.tools.scoring import DistanceToTheBest
from zellij.strategies.tools.tree_search import MoveUp
from zellij.utils.converters    import FloatMinMax, ArrayConverter, DoNothing

lf = Loss( objective=["obj"])(himmelblau)
values = ArrayVar(
    FloatVar("float_1", -5 , 5, converter=FloatMinMax()),
    FloatVar("float_2", -5, 5, converter=FloatMinMax()),
    converter=ArrayConverter(),
)
sp = Hypersphere(values, lf)

explor = PHS(sp, inflation=1.75)
exploi = ILS(sp, inflation=1.75)
stop1 = Threshold(None, "current_calls", 3)  # set target to None, DBA will automatically asign it.
stop2 = Threshold(None,"current_calls", 100)  # set target to None, DBA will automatically asign it.
dba = DBA(sp, MoveUp(sp,5),(explor,stop1), (exploi,stop2),scoring=DistanceToTheBest())

stop3 = Threshold(lf, "calls",1000)
exp = Experiment(dba, stop3, save="exp_fda")
exp.run()
print(f"Best solution:f({lf.best_point})={lf.best_score}")



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("exp_direct/outputs/all_evaluations.csv")
print(data)

fig, ax = plt.subplots()
x = y = np.linspace(-5, 5, 100)
X,Y = np.meshgrid(x,y)
Z = (X ** 2 + Y - 11) ** 2 + (X + Y ** 2 - 7) ** 2


map = ax.contourf(X,Y,Z,cmap="plasma", levels=100)
fig.colorbar(map)

plt.scatter(data["float_1"],data["float_2"],c="cyan",s=0.1)
plt.plot()