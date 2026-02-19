#!/bin/bash

#SBATCH --job-name="11-20"
#SBATCH --time=80:00:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=research-as-cheme


srun python ./11_pre_cah_140_100.py > ./11_pre_cah_140_100.log
srun python ./12_pre_cah_120_110.py > ./12_pre_cah_120_110.log
srun python ./13_pre_cah_130_110.py > ./13_pre_cah_130_110.log
srun python ./14_pre_cah_140_110.py > ./14_pre_cah_140_110.log
srun python ./15_pre_cah_130_120.py > ./15_pre_cah_130_120.log
srun python ./16_pre_cah_140_120.py > ./16_pre_cah_140_120.log
srun python ./17_pre_cah_140_130.py > ./17_pre_cah_140_130.log
srun python ./18_pre_cah_108_102.py > ./18_pre_cah_108_102.log
srun python ./19_pre_cah_115_95.py > ./19_pre_cah_115_95.log
srun python ./20_pre_cah_125_85.py > ./20_pre_cah_125_85.log
