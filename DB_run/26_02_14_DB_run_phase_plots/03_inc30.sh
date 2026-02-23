#!/bin/bash

#SBATCH --job-name="03_inc30"
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute-p1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./03_inc30.py > ./03_inc30.log 2>&1
