#!/bin/bash

# Submit all cook_spinach training jobs with different temporal_embedding_dim values
echo "Submitting cook_spinach training jobs with different temporal_embedding_dim values..."

echo "Submitting job for temporal_embedding_dim=128..."
sbatch train-cook-spinach-tdim128.slurm

echo "Submitting job for temporal_embedding_dim=64..."
sbatch train-cook-spinach-tdim64.slurm

echo "Submitting job for temporal_embedding_dim=32..."
sbatch train-cook-spinach-tdim32.slurm

echo "Submitting job for temporal_embedding_dim=16..."
sbatch train-cook-spinach-tdim16.slurm

echo "Submitting job for temporal_embedding_dim=8..."
sbatch train-cook-spinach-tdim8.slurm

echo "All cook_spinach temporal embedding jobs submitted!"
echo "Jobs will run on either NvidiaAll or Abaki partitions depending on availability"
echo "Use 'squeue -u \$USER' to check job status"
echo "Use 'scancel <job_id>' to cancel a specific job if needed" 