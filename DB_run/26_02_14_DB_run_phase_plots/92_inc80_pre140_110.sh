#!/bin/bash

#SBATCH --job-name="92_inc80_pre140_110"
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute-p1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./92_inc80_pre140_110.py > ./92_inc80_pre140_110.log 2>&1
