from hpo.core.searchspace import SearchSpace, Solution
from algorithm.soo import SOO

def himmelblau(solution : Solution) :
        x = solution.get_values()
        y = (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
        solution.add_score(y)
        return y

space_dict = {
    "x1" : {"min" : -5,"max" : 5, "type":"float"},
    "x2" : {"min" : -5,"max" : 5,"type":"float"}
}
space = SearchSpace(
    space_dict)
algo = SOO(search_space=space,
            K=3,
            maximizer=False,
            objective_function=himmelblau)
algo.run(budget=50)
algo.print()