#!/bin/bash

# Submit all vrig-chicken training jobs with different gaussian_embedding_dim values
echo "Submitting vrig-chicken training jobs with different gaussian_embedding_dim values..."

echo "Submitting job for gaussian_embedding_dim=4..."
sbatch train-vrig-chicken-gdim4.slurm

echo "Submitting job for gaussian_embedding_dim=8..."
sbatch train-vrig-chicken-gdim8.slurm

echo "Submitting job for gaussian_embedding_dim=16..."
sbatch train-vrig-chicken-gdim16.slurm

echo "Submitting job for gaussian_embedding_dim=64..."
sbatch train-vrig-chicken-gdim64.slurm

echo "All jobs submitted!"
echo "Use 'squeue -u \$USER' to check job status"
echo "Use 'scancel <job_id>' to cancel a specific job if needed" 