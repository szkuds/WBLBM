#!/bin/bash

#SBATCH --job-name="00_chem_step_inc_40"
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute-p1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme

srun python ./00_chem_step_inc_40.py > ./00_chem_step_inc_40.log 2>&1
