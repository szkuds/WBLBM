#!/bin/bash

#SBATCH --job-name="43_inc70_pre130_90"
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute-p1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./43_inc70_pre130_90.py > ./43_inc70_pre130_90.log 2>&1
