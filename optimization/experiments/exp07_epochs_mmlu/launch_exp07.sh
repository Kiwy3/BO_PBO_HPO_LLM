#!/bin/bash
#OAR -q testing
#OAR -p chuc
#OAR -l walltime=15
#OAR -n epoch_sensitivity_llm_mmlu
#OAR -O exp07_%jobid%.out
#OAR -E exp07_%jobid%.err


hostname
date

cd /home/ndavouse/ft_poc
MAIN_FOLDER="optimization/experiments/exp07_epochs_mmlu"


source /home/ndavouse/ft_poc/.venv/bin/activate
python ./$MAIN_FOLDER/exp07.py

CONFIG_FILE=$(jq -r '.experiment.historic_file' $MAIN_FOLDER/config.json)

# Check if the backup name was retrieved successfully
if [ -z "$BACKUP_NAME" ]; then
    echo "Error: Could not read backup_name from config.json"
    exit 1
fi

# Perform the backup
cp $CONFIG_FILE "./backup.json"
echo "Backup created: backup.json"