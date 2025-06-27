#!/bin/bash
#SBATCH --job-name=vrig-chicken-tdim16
#SBATCH --partition=NvidiaAll,Abaki
#SBATCH --output=slurm_logs/vrig-chicken-tdim16_%j.out
#SBATCH --error=slurm_logs/vrig-chicken-tdim16_%j.err

# Proper conda initialization
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate ed3dgs

python train.py -s data/vrig-chicken --configs arguments/hypernerf/vrig-chicken-tdim16.py --model_path output/hypernerf/vrig-chicken-tdim16 --expname hypernerf/vrig-chicken-tdim16 -r 2

echo "Job completed" 