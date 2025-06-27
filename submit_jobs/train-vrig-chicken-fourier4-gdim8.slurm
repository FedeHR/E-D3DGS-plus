#!/bin/bash
#SBATCH --job-name=vrig-chicken-fourier4-gdim8
#SBATCH --partition=NvidiaAll
#SBATCH --output=slurm_logs/vrig-chicken-fourier4-gdim8_%j.out
#SBATCH --error=slurm_logs/vrig-chicken-fourier4-gdim8_%j.err

# Proper conda initialization
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate ed3dgs

python train.py -s data/vrig-chicken --configs arguments/hypernerf/vrig-chicken-fourier4-gdim8.py --model_path output/hypernerf/vrig-chicken-fourier4-gdim8 --expname hypernerf/vrig-chicken-fourier4-gdim8 -r 2
echo "Job completed" 