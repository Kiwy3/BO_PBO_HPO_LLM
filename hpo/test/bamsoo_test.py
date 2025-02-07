from hpo.core.searchspace import SearchSpace, Solution
from hpo.algorithm.bamsoo_2 import BaMSOO

def himmelblau(solution : Solution) :
        x = solution.get_values()
        y = (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
        return y

space_dict = {
    "x1" : {"min" : -5,"max" : 5, "type":"float"},
    "x2" : {"min" : -5,"max" : 5,"type":"float"}
}
space = SearchSpace(
    space_dict)
algo = BaMSOO(search_space=space,
            K=3,
            maximizer=False,
            objective_function=himmelblau,
            eta = 1)
algo.run(budget=50)
algo.print()