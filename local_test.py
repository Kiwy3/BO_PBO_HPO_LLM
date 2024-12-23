from optimization_v2.toolbox import ModelEval, SearchSpace, Solution
from optimization_v2.algorithm.soo import SOO



space = SearchSpace(mode="base",
    savefile="savefile.json",)

evaluator = ModelEval(
    search_space = space,
    dev_run="fake"
)

soo = SOO(
    space=space,
    maximizer=True,
    K=3,
    obj_fun=evaluator.train_and_evaluate
)


soo.run(
    budget=5,
    saving=False
)

soo.print()

""" out = evaluator.train_and_evaluate(space.get_center())
spaces = space.section(3)

sp1 = spaces[0]
sp2 = spaces[1]

x1 = sp1.get_center()
x2 = sp2.get_center()

y1 = x1.get_values()
y2 = x2.get_values()

print(y1)
print(y2)

diff = 0.
for i in range(len(y1)):
    diff += abs(y1[i] - y2[i])
if diff < 0.0001 :
    print("same")
else :
    print("diff") """
