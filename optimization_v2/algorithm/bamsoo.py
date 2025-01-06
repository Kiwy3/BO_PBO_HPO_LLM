from optimization_v2.algorithm.soo import SOO, leaf, fun_error
from optimization_v2.toolbox.SearchSpace import SearchSpace, Solution

from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import math

from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

class BaMSOO(SOO):

    def __init__(self, #OK
            space : Optional[SearchSpace]  = None,
            maximizer : Optional[bool] = True,
            K : int = 3,
            filename : str = None,
            eta : float = 0.5,
            obj_fun = fun_error) :
        
        super().__init__( 
            space = space,
            maximizer = maximizer,
            K = K,
            filename = filename,
            obj_fun=obj_fun )
        self.eta = eta

    def get_points(self):
        points = [l.space.get_center(type="list") for l in self.tree.values() if l.score_state=="evaluated"]
        scores = [l.score for l in self.tree.values() if l.score_state=="evaluated"]
        return points, scores


    def update_gp(self):
        X,Y = self.get_points()
        train_X = torch.tensor(X).to(torch.double)
        train_Y = torch.tensor(Y).unsqueeze(-1).to(torch.double)
        gp = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=Normalize(d=train_X.shape[1]),
            outcome_transform=Standardize(m=train_Y.shape[1]),    
            )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        self.gp = gp

    def mean_sigma(self,X):
        posterior = self.gp.posterior(X)
        return posterior.mean.detach().item(),posterior.stddev.detach().item()
    
    def UCB(self,x) :
        mean, sigma = self.mean_sigma(torch.tensor([[x]],dtype=torch.double))
        N = len(self.tree)
        beta = math.sqrt(
            2*math.log(
                (math.pi**2*N**2)/
                (6*self.eta))
        )
    
        return mean + beta * sigma
        
    def LCB(self,x) :
        mean, sigma = self.mean_sigma(torch.tensor([[x]],dtype=torch.double))
        N = len(self.tree)
        beta = math.sqrt(
            2*math.log(
                (math.pi**2*N**2)/
                (6*self.eta))
        )
        return mean - beta * sigma
    
    def scoring(self, l):
        self.update_gp()
        print("\t\tUCB : ",self.UCB(l.space.get_center(type="list")))
        if self.UCB(l.space.get_center(type="list")) >= self.fp : 
            score,score_state = super().scoring(l)
            self.n_eval +=1
        else : 
            score = self.LCB(l.space.get_center(type="list"))
            score_state = "approximated"
        
        if score > self.fp:
            self.fp = score

        return score, score_state
    def add_leaf(self,space,depth,new_j,
                 parent : leaf=None,init=False) : #OK

        l = leaf(
            space=space,
            depth=depth,
            depth_id = new_j        
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
            score, score_state = super().scoring(l)
            self.n_eval +=1
        else :        
            score, score_state = self.scoring(l)
        l.score = score
        l.score_state = score_state       
        self.tree[depth,new_j] = l
    
    def initiate(self): #OK
            self.add_leaf(
                space=self.search_space,
                depth=0,
                new_j=0,
                init= True       
            )

    def save(self,filename="bamsoo.json"):
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
                        self.update_gp()
                        

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