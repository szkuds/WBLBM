#!/bin/bash

#SBATCH --job-name="08_chem_step_H10_Diffneg10"
#SBATCH --time=60:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme

srun python ./08_chem_step_H10_Diffneg10.py > ./08_chem_step_H10_Diffneg10.log
