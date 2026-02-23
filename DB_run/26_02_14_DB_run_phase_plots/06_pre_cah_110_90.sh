#!/bin/bash

#SBATCH --job-name="06_pre_cah_110_90"
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute-p1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./06_pre_cah_110_90.py > ./06_pre_cah_110_90.log 2>&1
