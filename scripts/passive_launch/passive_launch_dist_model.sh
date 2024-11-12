#!/bin/bash
#OAR -q testing
#OAR -p chuc
#OAR -l walltime=12
#OAR -n LLM eval
#OAR -O OAR_BO_%jobid%.out
#OAR -E OAR_BO_%jobid%.err


hostname
date


cd ft_poc
source /home/ndavouse/ft_poc/.venv/bin/activate
python ./Distributed/dist_bo.py
deactivate