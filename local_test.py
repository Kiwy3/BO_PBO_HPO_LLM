from optimization_v2.toolbox import ModelEval, SearchSpace, Solution
from optimization_v2.algorithm import BoGp


space = SearchSpace(mode="base",
    savefile="savefile.json",)

evaluator = ModelEval(
    search_space = space,
    dev_run="fake"
)

bo = BoGp(
    space=space,
    maximizer=True,
    obj_fun=evaluator.train_and_evaluate
)


bo.run(
    budget=50,
    init_budget=10
)

print("scores : ", bo.scores)

import matplotlib.pyplot as plt
plt.plot(bo.scores)
plt.show()

