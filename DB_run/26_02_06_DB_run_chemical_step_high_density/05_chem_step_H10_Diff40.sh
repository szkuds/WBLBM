#!/bin/bash

#SBATCH --job-name="05_chem_step_H10_Diff40"
#SBATCH --time=60:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme

srun python ./05_chem_step_H10_Diff40.py > ./05_chem_step_H10_Diff40.log
