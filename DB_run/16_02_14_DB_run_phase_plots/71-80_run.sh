#!/bin/bash

#SBATCH --job-name="71-80"
#SBATCH --time=80:00:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./71_inc30_pre120_110.py > ./71_inc30_pre120_110.log
srun python ./72_inc40_pre120_110.py > ./72_inc40_pre120_110.log
srun python ./73_inc50_pre120_110.py > ./73_inc50_pre120_110.log
srun python ./74_inc60_pre120_110.py > ./74_inc60_pre120_110.log
srun python ./75_inc70_pre120_110.py > ./75_inc70_pre120_110.log
srun python ./76_inc80_pre120_110.py > ./76_inc80_pre120_110.log
srun python ./77_inc10_pre130_110.py > ./77_inc10_pre130_110.log
srun python ./78_inc20_pre130_110.py > ./78_inc20_pre130_110.log
srun python ./79_inc30_pre130_110.py > ./79_inc30_pre130_110.log
srun python ./80_inc40_pre130_110.py > ./80_inc40_pre130_110.log
