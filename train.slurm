#!/bin/bash
#SBATCH --job-name=vrig-chicken-train
#SBATCH --output=slurm_logs/vrig-chicken-train_%j.out
#SBATCH --error=slurm_logs/vrig-chicken-train_%j.err

# Proper conda initialization
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate ed3dgs

python train.py -s data/vrig-chicken --configs arguments/hypernerf/vrig-chicken.py --model_path output/hypernerf/vrig-chicken --expname hypernerf/vrig-chicken -r 2

echo "Job completed" 