#!/bin/bash

#SBATCH --job-name="82_inc60_pre130_110"
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute-p1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./82_inc60_pre130_110.py > ./82_inc60_pre130_110.log 2>&1
