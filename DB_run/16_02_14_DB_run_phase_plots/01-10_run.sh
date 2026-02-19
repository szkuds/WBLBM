#!/bin/bash

#SBATCH --job-name="01-10"
#SBATCH --time=80:00:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme

srun python ./01_inc10.py > ./01_inc10.log
srun python ./02_inc20.py > ./02_inc20.log
srun python ./03_inc30.py > ./03_inc30.log
srun python ./04_inc50.py > ./04_inc50.log
srun python ./05_inc70.py > ./05_inc70.log
srun python ./06_pre_cah_110_90.py > ./06_pre_cah_110_90.log
srun python ./07_pre_cah_100_90.py > ./07_pre_cah_100_90.log
srun python ./08_pre_cah_130_90.py > ./08_pre_cah_130_90.log
srun python ./09_pre_cah_140_90.py > ./09_pre_cah_140_90.log
srun python ./10_pre_cah_130_100.py > ./10_pre_cah_130_100.log
