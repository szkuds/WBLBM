#!/bin/bash

#SBATCH --job-name="124_inc80_pre108_102"
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute-p1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./124_inc80_pre108_102.py > ./124_inc80_pre108_102.log 2>&1
