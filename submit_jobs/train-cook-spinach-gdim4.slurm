#!/bin/bash
#SBATCH --job-name=cook-spinach-gdim4
#SBATCH --partition=NvidiaAll
#SBATCH --output=slurm_logs/cook-spinach-gdim4_%j.out
#SBATCH --error=slurm_logs/cook-spinach-gdim4_%j.err

# Proper conda initialization
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate ed3dgs

python train.py -s data/cook_spinach --configs arguments/dynerf/cook-spinach-gdim4.py --model_path output/dynerf/cook-spinach-gdim4 --expname dynerf/cook-spinach-gdim4 -r 2

echo "Job completed" 