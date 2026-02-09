#!/bin/bash

#SBATCH --job-name="06_chem_step_H10_Diff10"
#SBATCH --time=60:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme

srun python ./06_chem_step_H10_Diff10.py > ./06_chem_step_H10_Diff10.log
