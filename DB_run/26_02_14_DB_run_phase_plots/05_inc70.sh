#!/bin/bash

#SBATCH --job-name="05_inc70"
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute-p1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./05_inc70.py > ./05_inc70.log 2>&1
