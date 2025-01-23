import numpy as np
from copy import deepcopy as dc
import gc
import json
import matplotlib.pyplot as plt
from hpo.core.SearchSpace import SearchSpace, Solution
from typing import Dict, List, Literal, Optional, Tuple, Union

def himmelblau(x) :
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def fun_error():
    print("no function provided")

class SOO :
    def __init__(self, #OK
                 space : Optional[SearchSpace]  = None,
                 maximizer : Optional[bool] = True,
                 K : int = 3,
                 filename : str = None,
                 obj_fun = fun_error) :
        
        if filename is not None :
            self.load_from_file(filename)
        else : 
            self.tree = {} 
            self.search_space = space
            self.K = K
            self.maximizer = maximizer
            self.loop = 0
            self.n_eval = 0
            self.last_depth = 0
        self.objective = obj_fun

    def scoring(self,
                l) -> Tuple[float, Literal["evaluated","inherited","approximated"]] :
        x = l.space.get_center()
        l.score_state = "evaluated"
        x.info = {
            "depth" : l.depth,
            "depth_id" : l.depth_id,
            "loop" : l.loop,
            "score_state" : l.score_state
        }

        y = self.objective(x) * (-1 if not self.maximizer else 1)
        self.n_eval +=1
        return y, "evaluated"

    def __compare_center__(self,
                           x1 : Solution,
                           x2 : Solution) -> bool : #OK
        y1 = x1.get_values()
        y2 = x2.get_values()
        diff = 0.
        for i in range(len(y1)):
            diff += abs(y1[i] - y2[i])
        if diff < 0.0001 :
            print("same center")
            return True
        else :
            print("different center")
            return False
        
    def max_depth (self) : #OK
        depths = [key[0] for key in self.tree.keys()]
        return max(depths)+1

    def initiate(self): #OK
        self.add_leaf(
            space=self.search_space,
            depth=0,
            new_j=0          
        )

    def add_leaf(self,
                 space : SearchSpace,
                 depth : int,
                 new_j : int,
                 parent =None) : #OK
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

        score,score_state = self.scoring(l)
        l.score = score
        l.score_state = score_state
        
        self.tree[depth,new_j] = l

    def select(self,depth) : # OK

        filtered_leaves = {key[1]: l.score for key, l in self.tree.items() 
                           if (key[0] == depth and l.state)}
        if len(filtered_leaves)== 0 : return None
        filtered_scores = list(filtered_leaves.values())
        filtered_index = list(filtered_leaves.keys())
        max_index = np.argmax(filtered_scores)
        return filtered_index[max_index]
    
    def bestof(self): #OK
        leaves_score = [l.score for l in self.tree.values()]
        max_index = np.argmax(leaves_score)
        max_leaf = [l for l in self.tree.values()][max_index]
        print("best leaf : ",
              "\n \t best score : ", max_leaf.score,
              "\n \t center : ", max_leaf.space.get_center().get_values())
        return max_leaf

    def print(self) : #OK
        for depth in range(self.max_depth()):
            print(f"depth = {depth} ")
            depth_leaves = [l for key, l in self.tree.items() 
                           if key[0] == depth]
            for l in depth_leaves:
                print(f"\t leaf number {l.depth_id} : ")
                print(f"\t\t center : {l.space.get_center().get_values()}, score = {l.score}, state : {l.state}, score_state : {l.score_state}")

    def save(self, filename = "soo_tree.json"): #OK
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

        with open(filename, 'w') as f:
            json.dump(export, f,indent=2)

    def load_from_file(self, filename): #OK
        with open(filename, 'r') as f:
            data = json.load(f)

        # tree config
        tree_config = data['global']
        self.K = tree_config['K']
        self.maximizer = tree_config['maximizer']
        self.loop = tree_config['loop']
        self.n_eval = tree_config['n_eval']
        self.last_depth = tree_config['last_depth']
        self.search_space = SearchSpace(variables=tree_config['space'])

        
        leaves_data = data['tree']
        self.tree = {}
        for key in leaves_data.keys() :
            l = leaf(
                space=SearchSpace(variables=leaves_data[key]["space"]),
                depth=leaves_data[key]["depth"],
                loop=leaves_data[key]["loop"],
                depth_id=leaves_data[key]["depth_id"],
                score=leaves_data[key]["score"],
                score_state=leaves_data[key]["score_state"],
            )
            l.state = leaves_data[key]["state"]
            self.tree[l.depth,l.depth_id]=l

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

class leaf :
    def __init__(self,space : SearchSpace,
                 depth,
                 loop=0,
                 depth_id=0,
                 score = None,
                 score_state = "unknown") :
        self.global_id = str(depth) + "_" + str(depth_id)
        self.space = space
        self.depth = depth
        self.depth_id = depth_id
        self.score = score
        self.state= True
        self.loop = loop
        self.score_state = score_state