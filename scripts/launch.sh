#!/bin/bash
#OAR -q testing
#OAR -p chuc
#OAR -l walltime=50
#OAR -n soo_finetune_llm
#OAR -O exp10_%jobid%.out
#OAR -E exp10_%jobid%.err

date

source .env/bin/activate
python cli.py $1

deactivate

echo "Done at" 
date
