#!/bin/bash
#SBATCH --job-name=cook-spinach-tdim8
#SBATCH --partition=NvidiaAll,Abaki
#SBATCH --output=slurm_logs/cook-spinach-tdim8_%j.out
#SBATCH --error=slurm_logs/cook-spinach-tdim8_%j.err

# Proper conda initialization
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate ed3dgs

python train.py -s data/cook_spinach --configs arguments/dynerf/cook-spinach-tdim8.py --model_path output/dynerf/cook-spinach-tdim8 --expname dynerf/cook-spinach-tdim8 -r 2

echo "Job completed" 