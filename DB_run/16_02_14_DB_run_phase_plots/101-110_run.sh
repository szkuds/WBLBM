#!/bin/bash

#SBATCH --job-name="101-110"
#SBATCH --time=80:00:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./101_inc10_pre140_120.py > ./101_inc10_pre140_120.log
srun python ./102_inc20_pre140_120.py > ./102_inc20_pre140_120.log
srun python ./103_inc30_pre140_120.py > ./103_inc30_pre140_120.log
srun python ./104_inc40_pre140_120.py > ./104_inc40_pre140_120.log
srun python ./105_inc50_pre140_120.py > ./105_inc50_pre140_120.log
srun python ./106_inc60_pre140_120.py > ./106_inc60_pre140_120.log
srun python ./107_inc70_pre140_120.py > ./107_inc70_pre140_120.log
srun python ./108_inc80_pre140_120.py > ./108_inc80_pre140_120.log
srun python ./109_inc10_pre140_130.py > ./109_inc10_pre140_130.log
srun python ./110_inc20_pre140_130.py > ./110_inc20_pre140_130.log
