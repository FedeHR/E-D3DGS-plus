#!/bin/bash

# Submit all cook_spinach training jobs with different gaussian_embedding_dim values
echo "Submitting cook_spinach training jobs with different gaussian_embedding_dim values..."

echo "Submitting job for gaussian_embedding_dim=2..."
sbatch train-cook-spinach-gdim2.slurm

echo "Submitting job for gaussian_embedding_dim=4..."
sbatch train-cook-spinach-gdim4.slurm

echo "Submitting job for gaussian_embedding_dim=8..."
sbatch train-cook-spinach-gdim8.slurm

echo "Submitting job for gaussian_embedding_dim=16..."
sbatch train-cook-spinach-gdim16.slurm

echo "Submitting job for gaussian_embedding_dim=64..."
sbatch train-cook-spinach-gdim64.slurm

echo "All cook_spinach gaussian embedding jobs submitted!"
echo "Jobs will run on NvidiaAll partition"
echo "Use 'squeue -u \$USER' to check job status"
echo "Use 'scancel <job_id>' to cancel a specific job if needed" 