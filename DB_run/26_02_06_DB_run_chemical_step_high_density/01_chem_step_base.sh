#!/bin/bash

#SBATCH --job-name="01_chem_step_base"
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute-p1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme

srun python ./01_chem_step_base.py > ./01_chem_step_base.log 2>&1
