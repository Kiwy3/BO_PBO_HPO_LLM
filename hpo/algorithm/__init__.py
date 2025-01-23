#from optimization_v2.algorithm.bo_class import BO_HPO
from hpo.algorithm.soo import SOO
from hpo.algorithm.bamsoo import BaMSOO
from hpo.algorithm.bo import BoGp

__all__ = ["SOO",
        "BaMSOO",
        "BoGp"]
