import os
os.chdir("/home/jan/Documents/Nathan1/ST30_1/Fine_tuning/MVP")

from zellij.core.geometry import Soo
from zellij.strategies import DBA
from zellij.strategies.tools.tree_search import Soo_tree_search
from zellij.strategies.tools.scoring import Distance_to_the_best_corrected

from zellij.core import ContinuousSearchspace, FloatVar, ArrayVar,IntVar, Loss
from zellij.utils.benchmarks import himmelblau

from zellij.core.objective import Maximizer,Minimizer
import matplotlib.pyplot as plt
import numpy as np

from black_box_llm import loss_function

Bounds = [[2,16],
          [8,16],
          [-10,-1]]

loss = Loss(objective=Minimizer)(loss_function)
values = ArrayVar(
                  FloatVar("loRA",Bounds[0][0],Bounds[0][1]),
                  FloatVar("Alpha",Bounds[1][0],Bounds[1][1]),
                  FloatVar("rate",Bounds[2][0],Bounds[2][1]),
                  )

def SOO_al(
  values,
  loss,
  calls,
  verbose=True,
  level=600,
  section=3,
  force_convert=False,
  ):

  sp = Soo(
      values,
      loss,
      calls,
      force_convert=force_convert,
      section=section,
  )

  dba = DBA(
      sp,
      calls,
      tree_search=Soo_tree_search(sp, level),
      verbose=verbose,
  )
  dba.run()

  return sp

sp = SOO_al(values, loss, 500)
best = (sp.loss.best_point, sp.loss.best_score)
print(f"Best solution found:f({best[0]})={best[1]}")