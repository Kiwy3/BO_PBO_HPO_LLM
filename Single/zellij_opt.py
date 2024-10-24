from zellij.core.geometry import Direct
from zellij.strategies import DBA
from zellij.strategies.tools.tree_search import Potentially_Optimal_Rectangle
from zellij.strategies.tools.direct_utils import Sigma2, SigmaInf

from zellij.core import ContinuousSearchspace, FloatVar, ArrayVar, Loss
from zellij.utils.benchmarks import himmelblau

import math
from bb_llm import BB_eval

def loss_function(x):
    HP = {
        "learning_rate": math.exp(x[0]),
        "lora_rank": round(x[1]),
        "fast_run" : True
    }
    return BB_eval(HP)

loss = Loss()(loss_function)
values = ArrayVar(
                  FloatVar("learning_rate",-10,-1),
                  FloatVar("lora_rank",2,32)
                  )
                  

def Direct_al(
  values,
  loss,
  calls,
  verbose=True,
  level=600,
  error=1e-4,
  maxdiv=3000,
  force_convert=False,
):

  sp = Direct(
      values,
      loss,
      calls,
      sigma=Sigma2(len(values)),
  )

  dba = DBA(
      sp,
      calls,
          tree_search=Potentially_Optimal_Rectangle(
          sp, level, error=error, maxdiv=maxdiv
      ),
      verbose=verbose,
  )
  dba.run()

  return sp

sp = Direct_al(values, loss, 40)
best = (sp.loss.best_point, sp.loss.best_score)
print(f"Best solution found:f({best[0]})={best[1]}")
print("solutions",sp.loss.all_solutions)
print("scores",sp.loss.all_scores)