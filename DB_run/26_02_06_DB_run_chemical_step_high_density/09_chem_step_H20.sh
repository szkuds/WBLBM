#!/bin/bash

#SBATCH --job-name="09_chem_step_H20"
#SBATCH --time=60:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme

srun python ./09_chem_step_H20.py > ./09_chem_step_H20.log
