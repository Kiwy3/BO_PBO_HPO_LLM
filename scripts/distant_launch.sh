date

ssh lille.g5k
oarsub -S ft_poc/scripts/passive_launch/passive_launch_bo.sh
exit
scp g5k:lille/ft_poc/optimization/export.json   new_bo.json
