#!/bin/bash

#SBATCH --job-name="81-90"
#SBATCH --time=80:00:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./81_inc50_pre130_110.py > ./81_inc50_pre130_110.log
srun python ./82_inc60_pre130_110.py > ./82_inc60_pre130_110.log
srun python ./83_inc70_pre130_110.py > ./83_inc70_pre130_110.log
srun python ./84_inc80_pre130_110.py > ./84_inc80_pre130_110.log
srun python ./85_inc10_pre140_110.py > ./85_inc10_pre140_110.log
srun python ./86_inc20_pre140_110.py > ./86_inc20_pre140_110.log
srun python ./87_inc30_pre140_110.py > ./87_inc30_pre140_110.log
srun python ./88_inc40_pre140_110.py > ./88_inc40_pre140_110.log
srun python ./89_inc50_pre140_110.py > ./89_inc50_pre140_110.log
srun python ./90_inc60_pre140_110.py > ./90_inc60_pre140_110.log
