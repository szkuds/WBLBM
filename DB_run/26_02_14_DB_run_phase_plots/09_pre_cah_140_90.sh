#!/bin/bash

#SBATCH --job-name="09_pre_cah_140_90"
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute-p1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./09_pre_cah_140_90.py > ./09_pre_cah_140_90.log 2>&1
