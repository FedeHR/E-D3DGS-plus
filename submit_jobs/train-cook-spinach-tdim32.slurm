#!/bin/bash
#SBATCH --job-name=cook-spinach-tdim32
#SBATCH --partition=NvidiaAll,Abaki
#SBATCH --output=slurm_logs/cook-spinach-tdim32_%j.out
#SBATCH --error=slurm_logs/cook-spinach-tdim32_%j.err

# Proper conda initialization
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate ed3dgs

python train.py -s data/cook_spinach --configs arguments/dynerf/cook-spinach-tdim32.py --model_path output/dynerf/cook-spinach-tdim32 --expname dynerf/cook-spinach-tdim32 -r 2

echo "Job completed" 