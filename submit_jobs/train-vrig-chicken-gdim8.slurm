#!/bin/bash
#SBATCH --job-name=vrig-chicken-gdim8
#SBATCH --partition=NvidiaAll
#SBATCH --output=slurm_logs/vrig-chicken-gdim8_%j.out
#SBATCH --error=slurm_logs/vrig-chicken-gdim8_%j.err

# Proper conda initialization
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate ed3dgs

python train.py -s data/vrig-chicken --configs arguments/hypernerf/vrig-chicken-gdim8.py --model_path output/hypernerf/vrig-chicken-gdim8 --expname hypernerf/vrig-chicken-gdim8 -r 2

echo "Job completed" 