#!/bin/bash

#SBATCH --job-name="121-130"
#SBATCH --time=80:00:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./121_inc50_pre108_102.py > ./121_inc50_pre108_102.log
srun python ./122_inc60_pre108_102.py > ./122_inc60_pre108_102.log
srun python ./123_inc70_pre108_102.py > ./123_inc70_pre108_102.log
srun python ./124_inc80_pre108_102.py > ./124_inc80_pre108_102.log
srun python ./125_inc10_pre115_95.py > ./125_inc10_pre115_95.log
srun python ./126_inc20_pre115_95.py > ./126_inc20_pre115_95.log
srun python ./127_inc30_pre115_95.py > ./127_inc30_pre115_95.log
srun python ./128_inc40_pre115_95.py > ./128_inc40_pre115_95.log
srun python ./129_inc50_pre115_95.py > ./129_inc50_pre115_95.log
srun python ./130_inc60_pre115_95.py > ./130_inc60_pre115_95.log
