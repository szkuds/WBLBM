#!/bin/bash

#SBATCH --job-name="134_inc20_pre125_85"
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute-p1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./134_inc20_pre125_85.py > ./134_inc20_pre125_85.log 2>&1
