#!/bin/bash
#OAR -q testing
#OAR -p chuc
#OAR -l walltime=5
#OAR -n Bayesian Opt of LLM
#OAR -O OAR_BO_%jobid%.out
#OAR -E OAR_BO_%jobid%.err


hostname
date


cd ft_poc
source /home/ndavouse/ft_poc/.venv/bin/activate
python ./bo.py