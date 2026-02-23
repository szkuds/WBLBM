#!/bin/bash

#SBATCH --job-name="118_inc20_pre108_102"
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute-p1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./118_inc20_pre108_102.py > ./118_inc20_pre108_102.log 2>&1
