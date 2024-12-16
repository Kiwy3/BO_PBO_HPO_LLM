#!/bin/bash
#OAR -q testing
#OAR -p chuc-1
#OAR -l walltime=15
#OAR -n extend_search_space
#OAR -O exp09_%jobid%.out
#OAR -E exp09_%jobid%.err


hostname
date

cd /home/ndavouse/ft_poc
MAIN_FOLDER="optimization/experiments/exp09_bo_space"


source /home/ndavouse/ft_poc/.venv/bin/activate
python ./$MAIN_FOLDER/exp05.py

CONFIG_FILE=$(jq -r '.experiment.historic_file' $MAIN_FOLDER/config.json)

# Check if the backup name was retrieved successfully
if [ -z "$BACKUP_NAME" ]; then
    echo "Error: Could not read backup_name from config.json"
    exit 1
fi

# Perform the backup
cp $CONFIG_FILE "./backup_exp09.json"
echo "Backup created: backup.json"