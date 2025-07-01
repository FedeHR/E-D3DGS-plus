#!/bin/bash

# E-D3DGS Log Organization Script
# Moves old logs without timestamp prefixes to archive directories

echo "🗂️ Organizing old SLURM logs..."

# Create archive directories
mkdir -p experiments/slurm_logs/archive/dynerf
mkdir -p experiments/slurm_logs/archive/hypernerf
mkdir -p experiments/slurm_logs/archive/technicolor

# Function to move old logs
move_old_logs() {
    local dataset=$1
    local source_dir="experiments/slurm_logs/$dataset"
    local archive_dir="experiments/slurm_logs/archive/$dataset"
    
    if [[ ! -d "$source_dir" ]]; then
        echo "   No $dataset directory found, skipping..."
        return
    fi
    
    echo "📂 Processing $dataset logs..."
    
    # Find directories that don't start with YYYYMMDD_HHMMSS pattern
    local moved_count=0
    for dir in "$source_dir"/*; do
        if [[ -d "$dir" ]]; then
            local dirname=$(basename "$dir")
            
            # Check if directory name starts with timestamp pattern (YYYYMMDD_HHMMSS)
            if [[ ! "$dirname" =~ ^[0-9]{8}_[0-9]{6}_ ]]; then
                echo "   Moving: $dirname → archive/$dataset/"
                mv "$dir" "$archive_dir/"
                ((moved_count++))
            fi
        fi
    done
    
    if [[ $moved_count -eq 0 ]]; then
        echo "   ✅ All $dataset logs already properly organized"
    else
        echo "   ✅ Moved $moved_count directories to archive"
    fi
}

# Process each dataset
move_old_logs "dynerf"
move_old_logs "hypernerf" 
move_old_logs "technicolor"

echo ""
echo "📋 Summary:"
echo "   Old logs moved to: experiments/slurm_logs/archive/"
echo "   New experiments will use timestamp-prefixed naming"
echo ""
echo "📅 To view organized logs:"
echo "   ls -t experiments/slurm_logs/dynerf/     # Newest first"
echo "   ls -t experiments/slurm_logs/hypernerf/  # Chronological order"
echo ""
echo "🎉 Log organization complete!" 