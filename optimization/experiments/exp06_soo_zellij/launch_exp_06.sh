#!/bin/bash
#OAR -q testing
#OAR -p chuc
#OAR -l walltime=15
#OAR -n soo_zellij
#OAR -O exp06_%jobid%.out
#OAR -E exp06_%jobid%.err


hostname
date

cd /home/ndavouse/ft_poc
MAIN_FOLDER="optimization/experiments/exp06_soo_zellij"


source /home/ndavouse/ft_poc/.venv_10/bin/activate
python ./$MAIN_FOLDER/exp06.py

CONFIG_FILE=$(jq -r '.experiment.historic_file' $MAIN_FOLDER/config.json)

# Check if the backup name was retrieved successfully
if [ -z "$BACKUP_NAME" ]; then
    echo "Error: Could not read backup_name from config.json"
    exit 1
fi

# Perform the backup
cp $CONFIG_FILE "./backup_exp06.json"
echo "Backup created: backup.json"