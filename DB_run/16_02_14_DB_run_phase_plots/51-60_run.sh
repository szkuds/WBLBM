#!/bin/bash

#SBATCH --job-name="51-60"
#SBATCH --time=80:00:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./51_inc70_pre140_90.py > ./51_inc70_pre140_90.log
srun python ./52_inc80_pre140_90.py > ./52_inc80_pre140_90.log
srun python ./53_inc10_pre130_100.py > ./53_inc10_pre130_100.log
srun python ./54_inc20_pre130_100.py > ./54_inc20_pre130_100.log
srun python ./55_inc30_pre130_100.py > ./55_inc30_pre130_100.log
srun python ./56_inc40_pre130_100.py > ./56_inc40_pre130_100.log
srun python ./57_inc50_pre130_100.py > ./57_inc50_pre130_100.log
srun python ./58_inc60_pre130_100.py > ./58_inc60_pre130_100.log
srun python ./59_inc70_pre130_100.py > ./59_inc70_pre130_100.log
srun python ./60_inc80_pre130_100.py > ./60_inc80_pre130_100.log
