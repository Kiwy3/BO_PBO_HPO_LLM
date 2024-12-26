from optimization_v2.toolbox import ModelEval, SearchSpace, Solution




space = SearchSpace(mode="base",
    savefile="savefile.json",)
spaces = space.section(3)

evaluator = ModelEval(
    search_space = space,
    dev_run="real"
)

""" results = []
for sp in spaces : 
    x = sp.get_center()
    out = evaluator.train_and_evaluate(x)
    results.append(out)
    print(out)


print("final results : ", results)
 """
results = []

x1 = space.get_solution(
    x=[32,1,0,0]
)
y1 = evaluator.train_and_evaluate(x1)
results.append(y1)

x2 = space.get_solution(
    x=[64,1,0,1]
)
y2 = evaluator.train_and_evaluate(x2)
results.append(y2)

x3 = space.get_solution(
    x=[8,1,0,2]
)
y3 = evaluator.train_and_evaluate(x3)
results.append(y3)

print("final results : ", results)