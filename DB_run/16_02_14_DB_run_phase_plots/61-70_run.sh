#!/bin/bash

#SBATCH --job-name="61-70"
#SBATCH --time=80:00:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./61_inc10_pre140_100.py > ./61_inc10_pre140_100.log
srun python ./62_inc20_pre140_100.py > ./62_inc20_pre140_100.log
srun python ./63_inc30_pre140_100.py > ./63_inc30_pre140_100.log
srun python ./64_inc40_pre140_100.py > ./64_inc40_pre140_100.log
srun python ./65_inc50_pre140_100.py > ./65_inc50_pre140_100.log
srun python ./66_inc60_pre140_100.py > ./66_inc60_pre140_100.log
srun python ./67_inc70_pre140_100.py > ./67_inc70_pre140_100.log
srun python ./68_inc80_pre140_100.py > ./68_inc80_pre140_100.log
srun python ./69_inc10_pre120_110.py > ./69_inc10_pre120_110.log
srun python ./70_inc20_pre120_110.py > ./70_inc20_pre120_110.log
