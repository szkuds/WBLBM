#!/bin/bash

#SBATCH --job-name="41-50"
#SBATCH --time=80:00:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./41_inc50_pre130_90.py > ./41_inc50_pre130_90.log
srun python ./42_inc60_pre130_90.py > ./42_inc60_pre130_90.log
srun python ./43_inc70_pre130_90.py > ./43_inc70_pre130_90.log
srun python ./44_inc80_pre130_90.py > ./44_inc80_pre130_90.log
srun python ./45_inc10_pre140_90.py > ./45_inc10_pre140_90.log
srun python ./46_inc20_pre140_90.py > ./46_inc20_pre140_90.log
srun python ./47_inc30_pre140_90.py > ./47_inc30_pre140_90.log
srun python ./48_inc40_pre140_90.py > ./48_inc40_pre140_90.log
srun python ./49_inc50_pre140_90.py > ./49_inc50_pre140_90.log
srun python ./50_inc60_pre140_90.py > ./50_inc60_pre140_90.log
