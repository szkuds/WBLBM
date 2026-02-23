#!/bin/bash

#SBATCH --job-name="10_pre_cah_130_100"
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute-p1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./10_pre_cah_130_100.py > ./10_pre_cah_130_100.log 2>&1
