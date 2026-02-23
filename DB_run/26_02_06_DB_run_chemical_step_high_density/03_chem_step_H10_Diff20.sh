#!/bin/bash

#SBATCH --job-name="03_chem_step_H10_Diff20"
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute-p1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme

srun python ./03_chem_step_H10_Diff20.py > ./03_chem_step_H10_Diff20.log 2>&1
