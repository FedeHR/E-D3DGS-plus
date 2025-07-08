#!/bin/bash

# Comprehensive Fourier Embedding Experiments for VRIG-Chicken Dataset
# Running 9 experiments: 3 embedding sizes (8, 16, 32) × 3 frequencies (4, 8, 16)
# Using --use_fourier_embedding_transform (not init) and no amplitude coefficients

echo "=========================================================================="
echo "Submitting Fourier Embedding Experiments for VRIG-Chicken Dataset"
echo "=========================================================================="
echo "Configurations:"
echo "  - Embedding sizes: 8, 16, 32"
echo "  - Frequencies: 4, 16" 
echo "  - Total experiments: 6"
echo "  - Partition: Abaki"
echo "  - No amplitude coefficients"
echo "  - Using lower opacity thresholds"
echo "=========================================================================="

# Create logs directory
mkdir -p slurm_logs

# Array to store job IDs
declare -a job_ids=()

# Define embedding sizes and frequencies
embedding_sizes=(8 16 32)
frequencies=(4 16)

# Submit jobs for each combination
for emb_size in "${embedding_sizes[@]}"; do
    for freq in "${frequencies[@]}"; do
        job_name="vrig-chicken-fourier-emb-${emb_size}d-${freq}freq-opacityth-0.001"
        
        echo "Submitting experiment: embedding_dim=${emb_size}, frequencies=${freq}"
        
        # Create individual SLURM script content
        cat > "temp_${job_name}.sh" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=Abaki
#SBATCH --qos=abaki
#SBATCH --time=6:00:00
#SBATCH --output=slurm_logs/${job_name}_%j.out
#SBATCH --error=slurm_logs/${job_name}_%j.err

# Load modules and activate environment
eval "\$(~/miniforge3/bin/conda shell.bash hook)"
conda activate ed3dgs

# Run training with Fourier embedding
python train.py \\
    -s data/vrig-chicken \\
    --model_path output/hypernerf/${job_name} \\
    --configs arguments/hypernerf/vrig-chicken.py \\
    --expname hypernerf/${job_name} \\
    --use_fourier_embedding_transform \\
    --gaussian_embedding_dim ${emb_size} \\
    --fourier_frequencies ${freq} \\
    -r 2

echo "Training completed for ${job_name}"
EOF

        # Submit the job and capture job ID
        job_id=$(sbatch "temp_${job_name}.sh" | awk '{print $4}')
        job_ids+=("${job_id}")
        
        echo "  Job ID: ${job_id}"
        
        # Clean up temporary script
        rm "temp_${job_name}.sh"
        
        # Small delay between submissions
        sleep 1
    done
done

echo ""
echo "=========================================================================="
echo "All experiments submitted successfully!"
echo "=========================================================================="
echo ""
echo "Job IDs:"
counter=0
for emb_size in "${embedding_sizes[@]}"; do
    for freq in "${frequencies[@]}"; do
        echo "  embedding_dim=${emb_size}, frequencies=${freq}: ${job_ids[$counter]}"
        ((counter++))
    done
done

echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: slurm_logs/"
echo ""
echo "Expected output directories:"
for emb_size in "${embedding_sizes[@]}"; do
    for freq in "${frequencies[@]}"; do
        echo "  output/hypernerf/vrig-chicken-fourier-emb-${emb_size}d-${freq}freq/"
    done
done