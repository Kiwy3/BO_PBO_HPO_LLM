# Import Zellij functions
from zellij.core.objective import Minimizer
from zellij.utils.benchmarks import himmelblau
from zellij.core import ArrayVar, FloatVar, Loss, Experiment, Threshold, BooleanStop
from zellij.strategies.fractals import CenterSOO, DBASampling
from zellij.strategies.tools import Section, Min
from zellij.strategies.tools.tree_search import SooTreeSearch
from zellij.utils.converters import FloatMinMax, ArrayDefaultC

# Import custom eval function
#from model_evaluation.model_eval import ModelEvaluator

lf = Loss( objective=[Minimizer("obj")])(himmelblau)
values = ArrayVar(
    FloatVar("float_1", -5 , 5, converter=FloatMinMax()),
    FloatVar("float_2", -5, 5, converter=FloatMinMax()),
    converter=ArrayDefaultC()
)
sp = Section(values, section=3)

explor = CenterSOO(sp)
#stop1 = BooleanStop(explor, "computed")  # set target to None, DBA will automatically asign it.
stop = Threshold(lf, "calls", 15)
dba = DBASampling(
    sp, SooTreeSearch(sp, float("inf")), explor, scoring=Min()
)
exp = Experiment(dba, lf, stop, 
                 save="soo"
                 )
exp.run()
print(f"Best solution:f({lf.best_point})={lf.best_score}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("soo/outputs/all_evaluations.csv")
print(df)

lower = -5
upper = 5
full_gap = upper - lower

fig, ax = plt.subplots(figsize=(8, 6))
x = y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = himmelblau([X, Y])["obj"]

map = ax.contourf(X, Y, Z, cmap="plasma", levels=100)
fig.colorbar(map)
df = df.iloc[:9]
plt.scatter(df["float_1"],df["float_2"],c="cyan",s=1)
for i in range(len(df)):
    plt.text(df["float_1"].iloc[i],df["float_2"].iloc[i],str(i))


plt.axvline(-5+10/3,color="red")
plt.axvline(-5+10/3*2,color="red")

for j in range(3,8,2):
    print("j = ",j)
    temp = df.iloc[j:j+2]
    if temp["float_1"].iloc[0] == temp["float_1"].iloc[1]:
        lb =  temp.iloc[temp["float_2"].argmin()]
        ub =  temp.iloc[temp["float_2"].argmax()]
        gap = ub.float_2 - lb.float_2
        if lb["fracid"] == 0:
            ratio = 1/3
        if lb.float_1 + ratio*full_gap > upper:
            plot_ub = 1
            plot_lb = 1-ratio
        elif lb.float_1 - ratio*full_gap < lower:
            plot_lb = 0
            plot_ub = ratio
        else:
            plot_lb = (1-ratio)/(2*full_gap)
            plot_ub = (1+ratio)/(2*full_gap)
        plt.axhline(lb.float_2+gap/4,plot_lb,plot_ub,color="red")
        plt.axhline(lb.float_2+3*gap/4,plot_lb,plot_ub,color="red")
    else :
        lb =  temp.iloc[temp["float_1"].argmin()]
        ub =  temp.iloc[temp["float_1"].argmax()]
        gap = ub.float_1 - lb.float_1
        middle = gap/2
        if lb["fracid"] == 0:
            ratio = 1/3
        elif lb["fracid"] == 3:
            ratio = 1/9
        if lb.float_2 + ratio*full_gap > upper:
            print("a")
            plot_ub = 1
            plot_lb = 1-ratio
        elif lb.float_2 - ratio*full_gap < lower:
            print("b")
            plot_lb = 0
            plot_ub = ratio
        else:
            print("c")
            plot_lb = 0.5 - 3/2*ratio
            plot_ub = 0.5 + 3/2*ratio
            print(lb.float_2,plot_lb,plot_ub)
        plt.axvline(lb.float_1+gap/4,plot_lb,plot_ub,color="red")
        plt.axvline(lb.float_1+3*gap/4,plot_lb,plot_ub,color="red")

plt.title("Simultaneous Optimistic Optimization")
plt.xlabel("dimension 1")
plt.ylabel("dimension 2")    
plt.savefig("plots/soo.jpg")

plt.show()



