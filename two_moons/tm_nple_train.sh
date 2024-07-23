#!/bin/zsh
source ~/miniforge3/bin/activate
conda activate cmpe

for ID in 0 1 2 3 4
do
python tm_nple_train.py --sim-budget=256 --threshold-step=800 --run-id=$ID
python tm_nple_train.py --sim-budget=512 --threshold-step=1600 --run-id=$ID
python tm_nple_train.py --sim-budget=1024 --threshold-step=3200 --run-id=$ID
python tm_nple_train.py --sim-budget=2048 --threshold-step=6400 --run-id=$ID
python tm_nple_train.py --sim-budget=4096 --threshold-step=12800 --run-id=$ID
done
