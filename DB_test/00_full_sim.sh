#!/bin/bash


#SBATCH --job-name="WBLBM_chemical_step_run"
#SBATCH --time=30:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./00_full_sim_chem_step.py > ./WBLBM_chemical_step_run.log
