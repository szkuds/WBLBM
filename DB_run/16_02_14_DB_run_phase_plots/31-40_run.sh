#!/bin/bash

#SBATCH --job-name="31-40"
#SBATCH --time=80:00:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./31_inc30_pre100_90.py > ./31_inc30_pre100_90.log
srun python ./32_inc40_pre100_90.py > ./32_inc40_pre100_90.log
srun python ./33_inc50_pre100_90.py > ./33_inc50_pre100_90.log
srun python ./34_inc60_pre100_90.py > ./34_inc60_pre100_90.log
srun python ./35_inc70_pre100_90.py > ./35_inc70_pre100_90.log
srun python ./36_inc80_pre100_90.py > ./36_inc80_pre100_90.log
srun python ./37_inc10_pre130_90.py > ./37_inc10_pre130_90.log
srun python ./38_inc20_pre130_90.py > ./38_inc20_pre130_90.log
srun python ./39_inc30_pre130_90.py > ./39_inc30_pre130_90.log
srun python ./40_inc40_pre130_90.py > ./40_inc40_pre130_90.log
