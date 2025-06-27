#!/bin/bash
#SBATCH --job-name=vrig-chicken-gdim32-baseline
#SBATCH --partition=NvidiaAll
#SBATCH --output=slurm_logs/vrig-chicken-gdim32-baseline_%j.out
#SBATCH --error=slurm_logs/vrig-chicken-gdim32-baseline_%j.err

# Proper conda initialization
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate ed3dgs

python train.py -s data/vrig-chicken --configs arguments/hypernerf/vrig-chicken-gdim32.py --model_path output/dynerf/vrig-chicken-gdim32-baseline --expname dynerf/vrig-chicken-baseline-gdim32-tdim256 -r 2

echo "Job completed" 