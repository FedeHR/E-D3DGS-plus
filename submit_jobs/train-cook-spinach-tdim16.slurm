#!/bin/bash
#SBATCH --job-name=cook-spinach-tdim16
#SBATCH --partition=NvidiaAll,Abaki
#SBATCH --output=slurm_logs/cook-spinach-tdim16_%j.out
#SBATCH --error=slurm_logs/cook-spinach-tdim16_%j.err

# Proper conda initialization
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate ed3dgs

python train.py -s data/cook_spinach --configs arguments/dynerf/cook-spinach-tdim16.py --model_path output/dynerf/cook-spinach-tdim16 --expname dynerf/cook-spinach-tdim16 -r 2

echo "Job completed" 