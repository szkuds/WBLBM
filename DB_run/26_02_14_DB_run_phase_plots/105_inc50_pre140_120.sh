#!/bin/bash

#SBATCH --job-name="105_inc50_pre140_120"
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute-p1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./105_inc50_pre140_120.py > ./105_inc50_pre140_120.log 2>&1
