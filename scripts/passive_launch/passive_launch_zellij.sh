#!/bin/bash
#OAR -q testing
#OAR -p chuc
#OAR -l walltime=20
#OAR -n Zellij_POC_LLM
#OAR -O OAR_zellij_%jobid%.out
#OAR -E OAR_zellij_%jobid%.err


hostname
date


cd ft_poc
source /home/ndavouse/ft_poc/.venv/bin/activate
python ./optimization/zellij_poc.py
deactivate