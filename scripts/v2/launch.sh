#!/bin/bash

date
cd /home/ndavouse/llm

source /home/ndavouse/llm/.env/bin/activate
python cli.py $1

deactivate

echo "Done at" 
date
