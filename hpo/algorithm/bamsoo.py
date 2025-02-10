from hpo.algorithm.soo import SOO, leaf
from hpo.algorithm.bo import BoGp
from hpo.core.searchspace import SearchSpace, Solution

from typing import Dict, List, Literal, Optional, Tuple, Union,Callable

import torch
import math

from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

class BaMSOO(SOO, BoGp):
    def __init__(self, #OK
            objective_function : Callable,
            search_space : Optional[SearchSpace]  = None,
            maximizer : Optional[bool] = True,
            K : int = 3,
            savefile : str = None,
            eta : float = 0.5,
            ) :
        
        SOO.__init__(self, 
            search_space = search_space,
            maximizer = maximizer,
            K = K,
            savefile = savefile,
            objective_function=objective_function)
        self.eta = eta

    def update_gp(self):
        train_X, train_Y = BoGp.get_points_tensors(self)
        gp = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=Normalize(d=train_X.shape[1]),
            outcome_transform=Standardize(m=train_Y.shape[1]),    
            )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        self.gp = gp

    def mean_sigma(self,
                   X:torch.Tensor) -> Tuple[float,float] :
        posterior = self.gp.posterior(X)
        return posterior.mean.detach().item(),posterior.stddev.detach().item()
    
    def UCB(self,
            x : List[float]) -> float:
        mean, sigma = self.mean_sigma(torch.tensor([[x]],dtype=torch.double))
        N = len(self.tree)
        beta = math.sqrt(
            2*math.log(
                (math.pi**2*N**2)/
                (6*self.eta))
        )
    
        return mean + beta * sigma
        
    def LCB(self,
            x : List[float]) -> float:
        mean, sigma = self.mean_sigma(torch.tensor([[x]],dtype=torch.double))
        N = len(self.tree)
        beta = math.sqrt(
            2*math.log(
                (math.pi**2*N**2)/
                (6*self.eta))
        )
        return mean - beta * sigma
    
    def scoring(self, 
                l:leaf)-> Tuple[float, Literal["evaluated","inherited","approximated"]] :
        self.update_gp()
        x = l.space.get_center()

        print("\t\tUCB : ",self.UCB(l.space.get_center(type="list")))
        print("\t\tLCB : ",self.LCB(l.space.get_center(type="list")))

        if self.UCB(l.space.get_center(type="list")) >= self.fp : 
            l.score_state = "evaluated"
            score = SOO.scoring(self,l)
        else : 
            score = self.LCB(l.space.get_center(type="list"))
            l.score_state = "approximated"
            x.add_score({"UCB" : self.UCB(l.space.get_center(type="list")),
                        "LCB" : self.LCB(l.space.get_center(type="list"))})

            # To save even approximated leaf
            x.info = {
                "depth" : l.depth,
                "depth_id" : l.depth_id,
                "loop" : l.loop,
                "score" : l.score,
                "score_state" : l.score_state}
            x.save()
        
        if score > self.fp:
            self.fp = score

        return score
    def add_leaf(self,space,depth,new_j,
                 parent : leaf=None,init=False) : #OK

        l = leaf(
            space=space,
            depth=depth,
            depth_id = new_j,
            loop=self.loop       
        )
        if parent is not None :
            if self.__compare_center__(parent.space.get_center(),l.space.get_center()) and parent.score_state in["evaluated","inherited"] : 
                l.score = parent.score
                l.score_state = "inherited"
                self.tree[depth,new_j] = l
                return 
            else : 
                parent.state=False
        
        if init : 
            score = SOO.scoring(self,l)
        else :        
            score = self.scoring(l)
        l.score = score      
        self.tree[depth,new_j] = l
    
    def initiate(self): #OK
            self.add_leaf(
                space=self.search_space,
                depth=0,
                new_j=0,
                init= True       
            )

    def save(self,filename="bamsoo_tree.json"):
        super().save(filename)

    def run(self,budget = 5,saving=False) :
        if self.n_eval == 0 : self.initiate();print("init done")
        self.fp = self.tree[0,0].score

        while self.n_eval <= budget :
            print("loop number : ", self.loop,"n_eval = ",self.n_eval, "f+ = ",self.fp)
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