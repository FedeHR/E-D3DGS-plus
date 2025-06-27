#!/bin/bash

# Submit all vrig-chicken training jobs with different temporal_embedding_dim values
echo "Submitting vrig-chicken training jobs with different temporal_embedding_dim values..."

echo "Submitting job for temporal_embedding_dim=128..."
sbatch train-vrig-chicken-tdim128.slurm

echo "Submitting job for temporal_embedding_dim=64..."
sbatch train-vrig-chicken-tdim64.slurm

echo "Submitting job for temporal_embedding_dim=32..."
sbatch train-vrig-chicken-tdim32.slurm

echo "Submitting job for temporal_embedding_dim=16..."
sbatch train-vrig-chicken-tdim16.slurm

echo "All temporal embedding jobs submitted!"
echo "Jobs will run on either Nvidia or Abaki partitions depending on availability"
echo "Use 'squeue -u \$USER' to check job status"
echo "Use 'scancel <job_id>' to cancel a specific job if needed" 
