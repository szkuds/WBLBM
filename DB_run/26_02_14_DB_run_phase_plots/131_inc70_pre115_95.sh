#!/bin/bash

#SBATCH --job-name="131_inc70_pre115_95"
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute-p1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./131_inc70_pre115_95.py > ./131_inc70_pre115_95.log 2>&1
