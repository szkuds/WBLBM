#!/bin/bash

#SBATCH --job-name="131-140"
#SBATCH --time=80:00:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./131_inc70_pre115_95.py > ./131_inc70_pre115_95.log
srun python ./132_inc80_pre115_95.py > ./132_inc80_pre115_95.log
srun python ./133_inc10_pre125_85.py > ./133_inc10_pre125_85.log
srun python ./134_inc20_pre125_85.py > ./134_inc20_pre125_85.log
srun python ./135_inc30_pre125_85.py > ./135_inc30_pre125_85.log
srun python ./136_inc40_pre125_85.py > ./136_inc40_pre125_85.log
srun python ./137_inc50_pre125_85.py > ./137_inc50_pre125_85.log
srun python ./138_inc60_pre125_85.py > ./138_inc60_pre125_85.log
srun python ./139_inc70_pre125_85.py > ./139_inc70_pre125_85.log
srun python ./140_inc80_pre125_85.py > ./140_inc80_pre125_85.log
