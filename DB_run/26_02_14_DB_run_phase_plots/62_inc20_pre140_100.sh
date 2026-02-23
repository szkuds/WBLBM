#!/bin/bash

#SBATCH --job-name="62_inc20_pre140_100"
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute-p1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./62_inc20_pre140_100.py > ./62_inc20_pre140_100.log 2>&1
