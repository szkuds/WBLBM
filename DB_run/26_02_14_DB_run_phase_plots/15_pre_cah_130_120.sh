#!/bin/bash

#SBATCH --job-name="15_pre_cah_130_120"
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute-p1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./15_pre_cah_130_120.py > ./15_pre_cah_130_120.log 2>&1
