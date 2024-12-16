import numpy as np
from copy import deepcopy as dc
import gc
import json
import matplotlib.pyplot as plt


def himmelblau(x) :
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def fun_error():
    print("no function provided")

class SOO :
    def __init__(self, #OK
                 space = None,
                 maximizer = True,
                 K = 3,
                 filename = None,
                 obj_fun=fun_error) :
        
        if filename is not None :
            self.load_from_file(filename)
        else : 
            self.tree = {}
            self.search_space = space
            self.K = K
            self.maximizer = maximizer
            self.loop = 0
            self.n_eval = 0
            self.search_space.coef()
        self.objective = obj_fun

    def scoring(self,l):
        x = l.space.center
        y = self.objective(x) * (-1 if not self.maximizer else 1)
        gc.collect()
        return y, "evaluated"

    def __compare_center__(self,x1,x2) : #OK
        diff = 0.
        for i in range(len(x1)):
            diff += abs(x1[i] - x2[i])
        if diff < 0.0001 :
            return True
        else :
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

    def add_leaf(self,space,depth,new_j,parent=None) : #OK
        l = leaf(
            space=space,
            depth=depth,
            depth_id = new_j,
            loop=self.loop      
        )
        if parent is not None :
            if self.__compare_center__(parent.space.center,l.space.center) : 
                l.score = parent.score
                l.score_state = "inherited"
                self.tree[depth,new_j] = l
                return 
            else : 
                parent.state=False

        score,score_state = self.scoring(l)
        l.score = score
        l.score_state = score_state
        self.n_eval +=1
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
              "\n \t center : ", max_leaf.space.center)
        return max_leaf

    def print(self) : #OK
        for depth in range(self.max_depth()):
            print(f"depth = {depth} ")
            depth_leaves = [l for key, l in self.tree.items() 
                           if key[0] == depth]
            for l in depth_leaves:
                print(f"\t leaf number {l.depth_id} : ")
                print(f"\t\t center : {l.space.center}, score = {l.score}, state : {l.state}")

    def save(self): #OK
        export = {
            "global" : {
                "K" : self.K,
                "maximizer" : self.maximizer,
                "space" : self.search_space.variables,
                "loop" : self.loop,
                "n_eval" : self.n_eval,
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
                "space" : l.space.variables,
                "loop" : l.loop,
                "score_state" : l.score_state
            }
        export["tree"] = tree

        with open("tree.json", 'w') as f:
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
        self.search_space = array(tree_config['space'])

        
        leaves_data = data['tree']
        self.tree = {}
        for key in leaves_data.keys() :
            l = leaf(
                space=array(leaves_data[key]["space"]),
                depth=leaves_data[key]["depth"],
                loop=leaves_data[key]["loop"],
                depth_id=leaves_data[key]["depth_id"],
                score=leaves_data[key]["score"],
                score_state=leaves_data[key]["score_state"],
            )
            self.tree[l.depth,l.depth_id]=l

    def run(self,budget = 5,saving=False) : #OK
        if self.n_eval == 0 : self.initiate();print("init done")

        while self.n_eval <= budget :
            print("loop number : ", self.loop,"n_eval = ",self.n_eval)
            vmax = float("-inf")
            for h in range(self.max_depth()):

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
            self.loop = self.loop + 1
        return self.bestof()

class leaf :
    def __init__(self,space,depth,loop=0,depth_id=0,score = None,score_state = "unknown") :
        self.global_id = str(depth) + "_" + str(depth_id)
        self.space = space
        self.depth = depth
        self.depth_id = depth_id
        self.score = score
        self.state= True
        self.loop = loop
        self.score_state = score_state

class array :
    def __init__(self,variables) :
        self.variables = variables
        self.dimensions = len(variables)
        self.center = self.get_center()


    def get_center(self) :
        x = []
        for var in self.variables.keys() :
            x.append(
                (self.variables[var]["min"] + self.variables[var]["max"])/2)
        return x
    
    def coef(self):
        for key in self.variables.keys() :
            self.variables[key]["coef"]=self.variables[key]["max"] - self.variables[key]["min"]

    
    def section(self,K) :
        spaces = [dc(self) for _ in range(K)]
        width = {}
        for var in self.variables.keys():
            width[var] = (self.variables[var]["max"] - self.variables[var]["min"])/self.variables[var]["coef"]



        dim = np.argmax(list(width.values()))
        var = list(self.variables.keys())[dim]

        lower = self.variables[var]["min"]
        upper = self.variables[var]["max"]
        steps = (upper - lower)/K
        for i in range(K) :
            spaces[i].variables[var]["min"] = lower + i*steps
            spaces[i].variables[var]["max"] = lower + (i+1)*steps
            spaces[i].center = spaces[i].get_center()
        return spaces
        