#!/bin/bash

#SBATCH --job-name="115_inc70_pre140_130"
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute-p1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./115_inc70_pre140_130.py > ./115_inc70_pre140_130.log 2>&1
