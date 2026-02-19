#!/bin/bash

#SBATCH --job-name="21-30"
#SBATCH --time=80:00:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./21_inc10_pre110_90.py > ./21_inc10_pre110_90.log
srun python ./22_inc20_pre110_90.py > ./22_inc20_pre110_90.log
srun python ./23_inc30_pre110_90.py > ./23_inc30_pre110_90.log
srun python ./24_inc40_pre110_90.py > ./24_inc40_pre110_90.log
srun python ./25_inc50_pre110_90.py > ./25_inc50_pre110_90.log
srun python ./26_inc60_pre110_90.py > ./26_inc60_pre110_90.log
srun python ./27_inc70_pre110_90.py > ./27_inc70_pre110_90.log
srun python ./28_inc80_pre110_90.py > ./28_inc80_pre110_90.log
srun python ./29_inc10_pre100_90.py > ./29_inc10_pre100_90.log
srun python ./30_inc20_pre100_90.py > ./30_inc20_pre100_90.log
