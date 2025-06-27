#!/bin/bash
#SBATCH --job-name=cook-spinach-gdim16
#SBATCH --partition=NvidiaAll
#SBATCH --output=slurm_logs/cook-spinach-gdim16_%j.out
#SBATCH --error=slurm_logs/cook-spinach-gdim16_%j.err

# Proper conda initialization
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate ed3dgs

python train.py -s data/cook_spinach --configs arguments/dynerf/cook-spinach-gdim16.py --model_path output/dynerf/cook-spinach-gdim16 --expname dynerf/cook-spinach-gdim16 -r 2

echo "Job completed" 