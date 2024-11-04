from zellij.utils.benchmarks import Rastrigin
from zellij.core import (
    ArrayVar,
    FloatVar,
    Loss,
    Experiment,
    Threshold,
    Minimizer,
)
from zellij.strategies.fractals import DBASampling, CenterSOO
from zellij.strategies.fractals.bo.bamsoo import BaMSOO
from zellij.strategies.tools import Section, Min, SooTreeSearch
from zellij.utils.converters import FloatMinMax, ArrayDefaultC


dim = 2
calls = dim * 10**4

fun = Rastrigin()

lf = Loss(objective=[Minimizer("obj")])(fun)

values = ArrayVar(converter=ArrayDefaultC())
for i in range(dim):
    values.append(
        FloatVar(f"float_{i+1}", fun.lower - 3, fun.upper, converter=FloatMinMax())
    )

sp = Section(values, section=3)

explor = CenterSOO(sp)

stop2 = Threshold(lf, "calls", calls)
dba = BaMSOO(sp, sampling = explor, nu=0.5)
exp = Experiment(dba, lf, stop2, save="bamsoo")
exp.run()
print(f"Best solution:f({lf.best_point})={lf.best_score}>={fun.optimum*dim}")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def print_res(path):
    data = pd.read_csv(path, encoding="latin-1")
    argmin = data["obj"].argmin()
    print(f"MIN : {data['obj'].min()}")
    print(len(data[data["algorithm"] == "ILSLHS"]))
    fig, ax = plt.subplots(figsize=(8, 8))
    x = y = np.linspace(fun.lower - 3, fun.upper, 100)
    X, Y = np.meshgrid(x, y)
    # Z = (1-X)**2+100*(Y-X**2)**2  # Rosenbrock
    Z = (
        X**2 - 10 * np.cos(2 * np.pi * X) + Y**2 - 10 * np.cos(2 * np.pi * Y) + 20
    )  # Rastrigin
    # Z = -20*np.exp(-0.2*np.sqrt(0.5*(X**2+Y**2)))-np.exp(0.5*(np.cos(2*np.pi*X)+np.cos(2*np.pi*Y)))+20+np.exp(1)
    # Z = (
    #     (X**2 / 4000 + Y**2 / 4000)
    #     - (
    #         np.cos(X)* np.cos(Y / np.sqrt(2))
    #     )
    #     + 1
    # ) # Griewank

    map = ax.contourf(X, Y, Z, cmap="plasma", levels=100)
    fig.colorbar(map)

    xc = data["float_1"]
    yc = data["float_2"]
    xb = data["float_1"].iloc[argmin]
    yb = data["float_2"].iloc[argmin]

    plt.scatter(xc, yc, c="cyan", s=10)
    plt.scatter(fun.poptimum, fun.poptimum, c="red", alpha=0.5)
    plt.scatter(xb, yb, c="green")
    plt.plot()

print_res("soo/outputs/all_evaluations.csv")