#!/bin/bash

#SBATCH --job-name="91-100"
#SBATCH --time=80:00:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./91_inc70_pre140_110.py > ./91_inc70_pre140_110.log
srun python ./92_inc80_pre140_110.py > ./92_inc80_pre140_110.log
srun python ./93_inc10_pre130_120.py > ./93_inc10_pre130_120.log
srun python ./94_inc20_pre130_120.py > ./94_inc20_pre130_120.log
srun python ./95_inc30_pre130_120.py > ./95_inc30_pre130_120.log
srun python ./96_inc40_pre130_120.py > ./96_inc40_pre130_120.log
srun python ./97_inc50_pre130_120.py > ./97_inc50_pre130_120.log
srun python ./98_inc60_pre130_120.py > ./98_inc60_pre130_120.log
srun python ./99_inc70_pre130_120.py > ./99_inc70_pre130_120.log
srun python ./100_inc80_pre130_120.py > ./100_inc80_pre130_120.log
