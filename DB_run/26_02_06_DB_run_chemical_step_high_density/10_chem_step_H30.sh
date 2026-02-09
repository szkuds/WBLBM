#!/bin/bash

#SBATCH --job-name="10_chem_step_H30"
#SBATCH --time=60:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme

srun python ./10_chem_step_H30.py > ./10_chem_step_H30.log
