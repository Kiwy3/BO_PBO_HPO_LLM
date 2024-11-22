#!/bin/bash
#OAR -q testing
#OAR -p chuc
#OAR -l walltime=15
#OAR -n BO_LLM_FT
#OAR -O OAR_BO_%jobid%.out
#OAR -E OAR_BO_%jobid%.err


hostname
date


cd ft_poc
source /home/ndavouse/ft_poc/.venv/bin/activate
python ./optimization/run.py

CONFIG_FILE=$(jq -r '.experiment.historic_file' config.json)

# Check if the backup name was retrieved successfully
if [ -z "$BACKUP_NAME" ]; then
    echo "Error: Could not read backup_name from config.json"
    exit 1
fi

# Perform the backup
cp $CONFIG_FILE "./backup.json"
echo "Backup created: backup.json"