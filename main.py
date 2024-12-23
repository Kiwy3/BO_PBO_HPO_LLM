from optimization_v2.toolbox import ModelEval, SearchSpace, Solution




space = SearchSpace(mode="base",
    savefile="savefile.json",)
x = space.center

""" evaluator = ModelEval(
    search_space = space,
    dev_run="real"
)

out = evaluator.train_and_evaluate(x)

print("out : ",out)
 """


spaces = space.section(3)

for i,sp in enumerate(spaces): 
    print("space number : ",i)
    print("\t center : ", sp.get_center().get_values())
    for value in sp.variables.values():
        print("\t ",value.get_dict())
