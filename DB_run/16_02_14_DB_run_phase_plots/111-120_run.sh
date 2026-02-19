#!/bin/bash

#SBATCH --job-name="111-120"
#SBATCH --time=80:00:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./111_inc30_pre140_130.py > ./111_inc30_pre140_130.log
srun python ./112_inc40_pre140_130.py > ./112_inc40_pre140_130.log
srun python ./113_inc50_pre140_130.py > ./113_inc50_pre140_130.log
srun python ./114_inc60_pre140_130.py > ./114_inc60_pre140_130.log
srun python ./115_inc70_pre140_130.py > ./115_inc70_pre140_130.log
srun python ./116_inc80_pre140_130.py > ./116_inc80_pre140_130.log
srun python ./117_inc10_pre108_102.py > ./117_inc10_pre108_102.log
srun python ./118_inc20_pre108_102.py > ./118_inc20_pre108_102.log
srun python ./119_inc30_pre108_102.py > ./119_inc30_pre108_102.log
srun python ./120_inc40_pre108_102.py > ./120_inc40_pre108_102.log
