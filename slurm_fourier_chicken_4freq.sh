#!/bin/bash
#SBATCH --job-name=vrig-chicken-fourier-init-4freq
#SBATCH --partition=NvidiaAll
#SBATCH --output=slurm_logs/vrig-chicken-fourier-init-4freq_%j.out
#SBATCH --error=slurm_logs/vrig-chicken-fourier-init-4freq_%j.err

# Load modules and activate environment
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate ed3dgs

# Run training with Fourier embedding initialization
python train.py \
    -s data/vrig-chicken \
    --model_path output/hypernerf/vrig-chicken-fourier-init-4freq \
    --configs arguments/hypernerf/vrig-chicken.py \
    --expname hypernerf/vrig-chicken-fourier-init-4freq --use_fourier_embedding_init \
    --fourier_frequencies 4 \
    -r 2

echo "Training completed for vrig-chicken-fourier-init-4freq" 