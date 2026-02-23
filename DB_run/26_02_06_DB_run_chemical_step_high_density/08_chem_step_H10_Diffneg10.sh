#!/bin/bash

#SBATCH --job-name="08_chem_step_H10_Diffneg10"
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute-p1
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme

srun python ./08_chem_step_H10_Diffneg10.py > ./08_chem_step_H10_Diffneg10.log 2>&1
