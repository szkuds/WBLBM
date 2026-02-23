#!/bin/bash

#SBATCH --job-name="32_inc40_pre100_90"
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute-p1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./32_inc40_pre100_90.py > ./32_inc40_pre100_90.log 2>&1
