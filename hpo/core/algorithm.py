from hpo.core.searchspace import SearchSpace, Solution
from typing import Dict, List, Literal, Optional, Tuple, Union, Callable
import numpy as np

class algorithm: 
    def __init__(self,
                 search_space : SearchSpace,
                 objective_function : Callable,
                 maximizer : Optional[bool] = True,
                 
                  ):
        self.n_eval = 0
        self.search_space = search_space
        self.maximizer = maximizer
        self.objective = objective_function
        self.historic : List[Solution] = []

    def scoring(self, 
                solution : Solution,
                info : Optional[Dict] = None) -> Tuple[Solution, float]:
        solution.info = info
        Y = self.objective(solution)*(-1 if not self.maximizer else 1)
        self.n_eval += 1
        if solution.score is None:
            solution.add_score(Y)
        self.historic.append(solution)
        return solution, Y
    
    def bestof(self) -> Solution:
        solution_score = [x.score for x in self.historic]
        max_index = np.argmax(solution_score)
        best_solution = self.historic[max_index]
        return best_solution
    
    def print(self):
        print("Printing all solutions")
        for i,x in enumerate(self.historic):
            print(f"Solution number {i+1}")
            print(f"\t Solution = {x.get_values()}")
            print(f"\t Score = {x.score}")
            print(f"\t Info = {x.info}")

    
    