#other part of hpo package
from hpo.core.SearchSpace import SearchSpace, Solution
from hpo.algorithm.optimization import algorithm
from hpo.algorithm.utils import leaf
#computation and file writing lib
import numpy as np
import json
#typing for documentation
from typing import Dict, List, Literal, Optional, Tuple, Union, Callable


class SOO(algorithm):
    def __init__(self,
                 search_space : SearchSpace,
                 objective_function : Callable,
                 maximizer : Optional[bool] = True,
                 savefile : Optional[str] = "soo_tree.json",
                 K : int = 3) :
        
        super().__init__(
            search_space = search_space,
            objective_function = objective_function,
            maximizer = maximizer
        )
        self.tree : dict[Tuple[int,int],leaf]  = {} 
        self.K = K
        self.savefile = savefile
        self.loop = 0
        self.last_depth = 0

    def scoring(self,
                l : leaf) -> Tuple[float, Literal["evaluated","inherited","approximated"]] :
        x = l.space.get_center()
        l.score_state = "evaluated"
        info = {
            "depth" : l.depth,
            "depth_id" : l.depth_id,
            "loop" : l.loop,
            "score_state" : l.score_state
        }
        X, Y = super().scoring(solution=x,info=info)
        return Y
    
    def __compare_center__(self,
                           x1 : Solution,
                           x2 : Solution) -> bool :
        y1 = x1.get_values()
        y2 = x2.get_values()
        diff = 0.
        for i in range(len(y1)):
            diff += abs(y1[i] - y2[i])
        if diff < 0.0001 :
            return True
        else :
            return False

    def max_depth (self) -> int:
        depths = [key[0] for key in self.tree.keys()]
        return max(depths)+1
    
    def initiate(self) -> None: 
        self.add_leaf(
            space=self.search_space,
            depth=0,
            new_j=0          
        )
    
    def add_leaf(self,
                 space : SearchSpace,
                 depth : int,
                 new_j : int,
                 parent : Optional[leaf] = None) :
        l = leaf(
            space=space,
            depth=depth,
            depth_id = new_j,
            loop=self.loop      
        )
        if parent is not None :
            if self.__compare_center__(parent.space.get_center(),l.space.get_center()) : 
                l.score = parent.score
                l.score_state = "inherited"
                self.tree[depth,new_j] = l
                return 
            else : 
                parent.state=False

        score = self.scoring(l)
        l.score = score
        
        self.tree[depth,new_j] = l

    def select(self,
               depth : int) -> int :

        filtered_leaves = {key[1]: l.score for key, l in self.tree.items() 
                           if (key[0] == depth and l.state)}
        if len(filtered_leaves)== 0 : return None
        filtered_scores = list(filtered_leaves.values())
        filtered_index = list(filtered_leaves.keys())
        max_index = np.argmax(filtered_scores)
        return filtered_index[max_index]
    
    def print(self) :
        for depth in range(self.max_depth()):
            print(f"depth = {depth} ")
            depth_leaves = [l for key, l in self.tree.items() 
                           if key[0] == depth]
            for l in depth_leaves:
                print(f"\t leaf number {l.depth_id} : ")
                print(f"\t\t center : {l.space.get_center().get_values()}, score = {l.score}, state : {l.state}, score_state : {l.score_state}")

    def save(self): #OK
        export = {
            "global" : {
                "K" : self.K,
                "maximizer" : self.maximizer,
                "space" : self.search_space.get_dict(),
                "loop" : self.loop,
                "n_eval" : self.n_eval,
                "last_depth" : self.last_depth
            },
        }
        tree = {}

        for key in self.tree.keys() :
            l = self.tree[key]
            tree[l.global_id] = {
                "global_id" : l.global_id,
                "depth" : l.depth,
                "depth_id" : l.depth_id,
                "score" : l.score,
                "state" : l.state,
                "space" : l.space.get_dict(),
                "loop" : l.loop,
                "score_state" : l.score_state
            }
        export["tree"] = tree

        with open(self.savefile, 'w') as f:
            json.dump(export, f,indent=2)

    def run(self,budget = 5,saving=False) : #OK
        if self.n_eval == 0 : self.initiate();print("init done")

        while self.n_eval < budget :
            print("loop number : ", self.loop,"n_eval = ",self.n_eval)
            vmax = float("-inf")
            for h in range(self.last_depth,self.max_depth()):
                self.last_depth = h

                j = self.select(h)
                print(f"\t h={h},j={j}")
                if j is None : continue

                if self.tree[h,j].score > vmax:
                    spaces = self.tree[h,j].space.section(self.K)
                    for i in range(self.K):
                        self.add_leaf(
                            parent = self.tree[h,j],
                            space = spaces[i],
                            depth= h+1,
                            new_j = self.K*j+i
                        )
                    vmax = self.tree[h,j].score
                if saving : self.save()
            self.last_depth = 0
            
            self.loop = self.loop + 1
        return self.bestof()