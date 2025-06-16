#!/bin/bash

# E-D3DGS Experiment Runner - Auto-detecting Dataset Types
# Usage: ./run_experiment.sh --scene SCENE_NAME [options]
# Example: ./run_experiment.sh --scene cut_roasted_beef --gdim 64

set -e  # Exit on any error

# Default parameters
SCENE=""
GDIM=32
TDIM=256
FOURIER_SCALE=0
NUM_FREQ_BANDS=""
EMBEDDING_INIT="zero"  # Changed to match current default
TEMPORAL_EMBEDDING_INIT="normal"
GPU=0
RESOLUTION=2
SLURM=true  # Default to SLURM mode
DRY_RUN=false
SKIP_RENDER=false
SKIP_EVAL=false

# Paths
GT_PATH="data"
SAVE_PATH="results"

# SLURM parameters - Fixed for abakus compatibility
SLURM_PARTITION="NvidiaAll"  # Default to NvidiaAll which works reliably
SLURM_QOS=""  # No QoS by default
SLURM_RESERVATION=""  # No reservation by default
SLURM_TIME="48:00:00"
SLURM_MEM=""  # No memory specification to avoid errors
SLURM_GPUS="1"
SLURM_CPUS="8"

# Dataset type mappings
declare -A DYNERF_SCENES=(
    ["coffee_martini"]="dynerf"
    ["cook_spinach"]="dynerf"
    ["cut_roasted_beef"]="dynerf"
    ["flame_salmon_1"]="dynerf"
    ["flame_steak"]="dynerf"
    ["sear_steak"]="dynerf"
)

declare -A HYPERNERF_SCENES=(
    ["aleks-teapot"]="hypernerf"
    ["chickchicken"]="hypernerf"
    ["cut-lemon"]="hypernerf"
    ["hand"]="hypernerf"
    ["slice-banana"]="hypernerf"
    ["torchocolate"]="hypernerf"
    ["americano"]="hypernerf"
    ["cross-hands"]="hypernerf"
    ["espresso"]="hypernerf"
    ["keyboard"]="hypernerf"
    ["oven-mitts"]="hypernerf"
    ["split-cookie"]="hypernerf"
    ["tamping"]="hypernerf"
    ["3dprinter"]="hypernerf"
    ["broom"]="hypernerf"
    ["vrig-chicken"]="hypernerf"
    ["peel-banana"]="hypernerf"
)

# Function to auto-detect dataset type
detect_dataset_type() {
    local scene_name="$1"
    
    # Check dynerf scenes
    if [[ -n "${DYNERF_SCENES[$scene_name]}" ]]; then
        echo "dynerf"
        return
    fi
    
    # Check hypernerf scenes (with and without prefixes)
    if [[ -n "${HYPERNERF_SCENES[$scene_name]}" ]]; then
        echo "hypernerf"
        return
    fi
    
    # Check for hypernerf scenes with prefixes
    for prefix in "interp_" "misc_" "vrig_"; do
        local base_name="${scene_name#$prefix}"
        if [[ -n "${HYPERNERF_SCENES[$base_name]}" ]]; then
            echo "hypernerf"
            return
        fi
    done
    
    # Check if scene exists in data directory
    if [[ -d "$GT_PATH/$scene_name" ]]; then
        # Try to detect based on directory structure
        if [[ -d "$GT_PATH/$scene_name/images" && -d "$GT_PATH/$scene_name/colmap" ]]; then
            echo "dynerf"  # Default to dynerf for standard structure
            return
        elif [[ -f "$GT_PATH/$scene_name/scene.json" ]]; then
            echo "hypernerf"  # HyperNeRF uses scene.json
            return
        fi
    fi
    
    # Default fallback
    echo "dynerf"
}

# Function to print usage
usage() {
    cat << EOF
🚀 E-D3DGS Experiment Runner (Auto-detecting Dataset Types)

Usage: $0 --scene SCENE_NAME [OPTIONS]

Required:
  --scene SCENE           Scene name (auto-detects dataset type)

Model Parameters:
  --gdim DIM              Gaussian embedding dimension (default: 32)
  --tdim DIM              Temporal embedding dimension (default: 256)
  --fourier_scale SCALE   Fourier features scale, 0=disabled (default: 0)
  --num_freq_bands N      Number of frequency bands for structured Fourier (optional)
  --embedding_init TYPE   Gaussian embedding initialization (default: zero)
                          Options: zero, random, normal, xavier, xavier_uniform, xavier_normal,
                                  kaiming, he_uniform, kaiming_normal, he_normal, uniform,
                                  fourier, positional, structured_fourier, learned_fourier
  --temporal_init TYPE    Temporal embedding initialization (default: normal)
                          Options: zero, normal, random, xavier_uniform, xavier_normal, sinusoidal

Training Options:
  --gpu GPU_ID            GPU to use (default: 0)
  --resolution RES        Resolution scaling factor (default: 2)
  --gt_path PATH          Path to datasets directory (default: data)

Execution Options:
  --slurm                 Submit to SLURM (default behavior)
  --no_slurm              Run locally instead of SLURM
  --dry_run               Show commands without executing
  --skip_render           Skip rendering step
  --skip_eval             Skip evaluation step

SLURM Options (only used with --slurm):
  --partition PART        SLURM partition (default: NvidiaAll)
  --abakus                Use Abaki partition with abaki QoS (auto-detects reservations)
  --reservation NAME      Manually specify reservation (compvis25_So, compvis25_Mo)
  --time TIME             SLURM time limit (default: 48:00:00)
  --cpus CPUS             SLURM CPUs (default: 8)

📋 Known Scenes:

DyNeRF Scenes (Neural 3D Video):
  - coffee_martini, cook_spinach, cut_roasted_beef
  - flame_salmon_1, flame_steak, sear_steak

HyperNeRF Scenes:
  - aleks-teapot, chickchicken, cut-lemon, hand, slice-banana, torchocolate
  - americano, cross-hands, espresso, keyboard, oven-mitts, split-cookie, tamping
  - 3dprinter, broom, vrig-chicken, peel-banana

Examples:
  # Simple experiment (auto-detects dynerf)
  $0 --scene cut_roasted_beef

  # Custom parameters
  $0 --scene cut_roasted_beef --gdim 64 --tdim 512

  # Enable Fourier features
  $0 --scene vrig-chicken --embedding_init fourier --fourier_scale 4.0

  # Use high-priority Abaki partition (auto-detects reservations)
  $0 --scene cut_roasted_beef --abakus

  # Use Abaki with specific reservation (Sunday/Monday)
  $0 --scene cut_roasted_beef --abakus --reservation compvis25_So

  # Run locally (no SLURM)
  $0 --scene cut_roasted_beef --no_slurm

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --scene)
            SCENE="$2"
            shift 2
            ;;
        --gt_path)
            GT_PATH="$2"
            shift 2
            ;;
        --gdim)
            GDIM="$2"
            shift 2
            ;;
        --tdim)
            TDIM="$2"
            shift 2
            ;;
        --fourier_scale)
            FOURIER_SCALE="$2"
            shift 2
            ;;
        --num_freq_bands)
            NUM_FREQ_BANDS="$2"
            shift 2
            ;;
        --embedding_init)
            EMBEDDING_INIT="$2"
            shift 2
            ;;
        --temporal_init)
            TEMPORAL_EMBEDDING_INIT="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        --slurm)
            SLURM=true
            shift
            ;;
        --no_slurm)
            SLURM=false
            shift
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        --skip_render)
            SKIP_RENDER=true
            shift
            ;;
        --skip_eval)
            SKIP_EVAL=true
            shift
            ;;
        --partition)
            SLURM_PARTITION="$2"
            shift 2
            ;;
        --abakus)
            SLURM_PARTITION="Abaki"
            SLURM_QOS="abaki"
            # Auto-detect reservation based on current day
            current_day=$(date +%u)  # 1=Monday, 7=Sunday
            if [[ "$current_day" == "7" ]]; then
                SLURM_RESERVATION="compvis25_So"
                echo "🎯 Sunday detected - using compvis25_So reservation"
            elif [[ "$current_day" == "1" ]]; then
                SLURM_RESERVATION="compvis25_Mo"
                echo "🎯 Monday detected - using compvis25_Mo reservation"
            else
                echo "⚠️  Outside reservation period (Sun/Mon) - using regular Abaki queue"
            fi
            shift
            ;;
        --reservation)
            SLURM_RESERVATION="$2"
            shift 2
            ;;
        --time)
            SLURM_TIME="$2"
            shift 2
            ;;
        --cpus)
            SLURM_CPUS="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$SCENE" ]]; then
    echo "❌ Error: --scene is required"
    echo ""
    usage
    exit 1
fi

# Auto-detect dataset type
DATASET=$(detect_dataset_type "$SCENE")

# Generate experiment name following the convention: dataset/scene-gdim32-tdim256-fourier4
generate_exp_name() {
    local exp_name="${DATASET}/${SCENE}"
    
    # Always add dimensions first
    local params="-gdim${GDIM}-tdim${TDIM}"
    
    # Add frequency bands for structured Fourier (if specified)
    if [[ -n "$NUM_FREQ_BANDS" ]]; then
        params="${params}-bands${NUM_FREQ_BANDS}"
    fi
    
    # Add Fourier features or initialization method at the end
    if [[ "$FOURIER_SCALE" != "0" ]]; then
        # Using Fourier features - add fourier scale at the end
        params="${params}-fourier${FOURIER_SCALE}"
    elif [[ "$EMBEDDING_INIT" != "zero" ]]; then
        # Not using Fourier but using non-default initialization
        params="${params}-${EMBEDDING_INIT}"
    fi
    
    # Add temporal initialization if non-default
    if [[ "$TEMPORAL_EMBEDDING_INIT" != "normal" ]]; then
        params="${params}-temporal${TEMPORAL_EMBEDDING_INIT}"
    fi
    
    echo "${exp_name}${params}"
}

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Generate experiment name
EXP_NAME=$(generate_exp_name)

# Create directory structure
mkdir -p experiments/slurm_jobs/$DATASET
mkdir -p experiments/slurm_logs/$DATASET

# Generate SLURM script filename
SLURM_SCRIPT="experiments/slurm_jobs/$DATASET/${DATASET}_${SCENE}${params}_${TIMESTAMP}.sh"

echo "🚀 E-D3DGS Experiment Runner"
echo "=============================="
echo "📋 Experiment Configuration:"
echo "   Scene: $SCENE"
echo "   Dataset: $DATASET (auto-detected)"
echo "   Experiment name: $EXP_NAME"
echo "   Gaussian embedding dim: $GDIM"
echo "   Temporal embedding dim: $TDIM"
echo "   Fourier scale: $FOURIER_SCALE $([ "$FOURIER_SCALE" = "0" ] && echo "(disabled - original behavior)" || echo "(enabled)")"
echo "   Embedding init: $EMBEDDING_INIT"
echo "   GPU: $GPU"
echo "   Resolution: $RESOLUTION"
echo "   SLURM: $SLURM"
if [[ "$SLURM" == "true" ]]; then
    echo "   SLURM partition: $SLURM_PARTITION"
    if [[ -n "$SLURM_QOS" ]]; then
        echo "   SLURM QoS: $SLURM_QOS"
    fi
    if [[ -n "$SLURM_RESERVATION" ]]; then
        echo "   SLURM reservation: $SLURM_RESERVATION"
    fi
fi
echo "   Dry run: $DRY_RUN"

if [[ "$SLURM" == "true" ]]; then
    echo ""
    echo "🔧 Generating SLURM script..."
    
    # Generate SLURM script
    cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${DATASET}_${SCENE}_${TIMESTAMP}
#SBATCH --partition=$SLURM_PARTITION
#SBATCH --time=$SLURM_TIME
#SBATCH --cpus-per-task=$SLURM_CPUS
#SBATCH --output=experiments/slurm_logs/$DATASET/${DATASET}_${SCENE}${params}_${TIMESTAMP}_%j.out
#SBATCH --error=experiments/slurm_logs/$DATASET/${DATASET}_${SCENE}${params}_${TIMESTAMP}_%j.err
EOF

    # Add memory specification only if provided
    if [[ -n "$SLURM_MEM" ]]; then
        echo "#SBATCH --mem=$SLURM_MEM" >> "$SLURM_SCRIPT"
    fi

    # Add QoS only if specified
    if [[ -n "$SLURM_QOS" ]]; then
        echo "#SBATCH --qos=$SLURM_QOS" >> "$SLURM_SCRIPT"
    fi

    # Add reservation only if specified
    if [[ -n "$SLURM_RESERVATION" ]]; then
        echo "#SBATCH --reservation=$SLURM_RESERVATION" >> "$SLURM_SCRIPT"
    fi

    cat >> "$SLURM_SCRIPT" << EOF

# === EXPERIMENT METADATA ===
# Scene: $SCENE
# Dataset: $DATASET (auto-detected)
# Gaussian Embedding Dim: $GDIM
# Temporal Embedding Dim: $TDIM
# Fourier Scale: $FOURIER_SCALE
# Experiment Name: $EXP_NAME
# Created: $TIMESTAMP
# ===========================

# Load environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ed3dgs

# Setup wandb environment
source bin/setup_wandb_team.sh

# Change to project directory
cd \$SLURM_SUBMIT_DIR

echo "🚀 E-D3DGS Experiment Starting"
echo "=============================="
echo "Job Name: ${DATASET}_${SCENE}_${TIMESTAMP}"
echo "Experiment: $EXP_NAME"
echo "Dataset: $DATASET (auto-detected)"
echo "Scene: $SCENE"
echo "Gaussian dim: $GDIM"
echo "Temporal dim: $TDIM"
echo "Fourier scale: $FOURIER_SCALE"
echo "SLURM Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURMD_NODENAME"
echo "Partition: \$SLURM_JOB_PARTITION"
echo "QoS: \$SLURM_JOB_QOS"
echo "GPU: \$CUDA_VISIBLE_DEVICES"
echo "Started: \$(date)"
echo "=============================="

# Create progress tracking file
echo "STARTED|\$(date)" > experiments/slurm_logs/$DATASET/${DATASET}_${SCENE}${params}_${TIMESTAMP}_progress.txt

# Training
echo "🚀 Starting training..."
echo "TRAINING|\$(date)" >> experiments/slurm_logs/$DATASET/${DATASET}_${SCENE}${params}_${TIMESTAMP}_progress.txt
CUDA_VISIBLE_DEVICES=$GPU python train.py -s $GT_PATH/$SCENE --port 0 --model_path $SAVE_PATH/$DATASET/${SCENE}_${DATASET}_${SCENE} --expname "$EXP_NAME" --configs arguments/$DATASET/$SCENE.py -r $RESOLUTION --embedding_init $EMBEDDING_INIT --temporal_embedding_init $TEMPORAL_EMBEDDING_INIT --fourier_scale $FOURIER_SCALE --gaussian_embedding_dim $GDIM --temporal_embedding_dim $TDIM$([ -n "$NUM_FREQ_BANDS" ] && echo " --num_freq_bands $NUM_FREQ_BANDS")

if [ \$? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "TRAINING_DONE|\$(date)" >> experiments/slurm_logs/$DATASET/${DATASET}_${SCENE}${params}_${TIMESTAMP}_progress.txt
    
EOF

    if [[ "$SKIP_RENDER" != "true" ]]; then
        cat >> "$SLURM_SCRIPT" << EOF
    # Rendering
    echo "🎨 Starting rendering..."
    echo "RENDERING|\$(date)" >> experiments/slurm_logs/$DATASET/${DATASET}_${SCENE}${params}_${TIMESTAMP}_progress.txt
    CUDA_VISIBLE_DEVICES=$GPU python render.py --model_path $SAVE_PATH/$DATASET/${SCENE}_${DATASET}_${SCENE} --skip_train --configs arguments/$DATASET/$SCENE.py
    
    if [ \$? -eq 0 ]; then
        echo "✅ Rendering completed successfully!"
        echo "RENDERING_DONE|\$(date)" >> experiments/slurm_logs/$DATASET/${DATASET}_${SCENE}${params}_${TIMESTAMP}_progress.txt
    else
        echo "❌ Rendering failed!"
        echo "RENDERING_FAILED|\$(date)" >> experiments/slurm_logs/$DATASET/${DATASET}_${SCENE}${params}_${TIMESTAMP}_progress.txt
        exit 1
    fi
    
EOF
    fi

    if [[ "$SKIP_EVAL" != "true" ]]; then
        cat >> "$SLURM_SCRIPT" << EOF
    # Evaluation
    echo "📈 Starting evaluation..."
    echo "EVALUATION|\$(date)" >> experiments/slurm_logs/$DATASET/${DATASET}_${SCENE}${params}_${TIMESTAMP}_progress.txt
    CUDA_VISIBLE_DEVICES=$GPU python metrics.py --model_path $SAVE_PATH/$DATASET/${SCENE}_${DATASET}_${SCENE}
    
    if [ \$? -eq 0 ]; then
        echo "✅ Evaluation completed successfully!"
        echo "COMPLETED|\$(date)" >> experiments/slurm_logs/$DATASET/${DATASET}_${SCENE}${params}_${TIMESTAMP}_progress.txt
    else
        echo "❌ Evaluation failed!"
        echo "EVALUATION_FAILED|\$(date)" >> experiments/slurm_logs/$DATASET/${DATASET}_${SCENE}${params}_${TIMESTAMP}_progress.txt
        exit 1
    fi
    
EOF
    fi

    cat >> "$SLURM_SCRIPT" << EOF
else
    echo "❌ Training failed!"
    echo "TRAINING_FAILED|\$(date)" >> experiments/slurm_logs/$DATASET/${DATASET}_${SCENE}${params}_${TIMESTAMP}_progress.txt
    exit 1
fi

echo "🎉 Experiment completed successfully!"
echo "Check your wandb dashboard: https://wandb.ai/\$WANDB_ENTITY/\$WANDB_PROJECT"
echo "Results saved to: $SAVE_PATH/$DATASET/${SCENE}_${DATASET}_${SCENE}"
echo "Completed: \$(date)"
EOF

    chmod +x "$SLURM_SCRIPT"
    echo "   Generated: $SLURM_SCRIPT"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo ""
        echo "🔍 Dry run mode - would execute:"
        echo "   sbatch $SLURM_SCRIPT"
        echo ""
        echo "📄 Generated SLURM script content:"
        echo "=================================="
        cat "$SLURM_SCRIPT"
    else
        echo "📤 Submitting to SLURM..."
        sbatch "$SLURM_SCRIPT"
        echo ""
        echo "✅ Job submitted successfully!"
        echo "📊 Monitor with: ./tools/monitor_experiments.sh --status"
        echo "📄 View logs with: ./tools/monitor_experiments.sh --logs"
        echo "👀 Watch live: ./tools/monitor_experiments.sh --watch"
    fi
else
    # Local execution
    echo ""
    echo "🖥️  Running locally..."
    echo "📍 This will run in your current terminal with real-time progress"
    echo "📊 You can monitor training progress directly in the terminal"
    echo "⚠️  Press Ctrl+C to stop the experiment"
    echo ""
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "🔍 Dry run mode - would execute:"
        echo "   CUDA_VISIBLE_DEVICES=$GPU python train.py -s $GT_PATH/$SCENE --port 0 --model_path $SAVE_PATH/$DATASET/${SCENE}_${DATASET}_${SCENE} --expname \"$EXP_NAME\" --configs arguments/$DATASET/$SCENE.py -r $RESOLUTION"
        if [[ "$SKIP_RENDER" != "true" ]]; then
            echo "   CUDA_VISIBLE_DEVICES=$GPU python render.py --model_path $SAVE_PATH/$DATASET/${SCENE}_${DATASET}_${SCENE} --skip_train --configs arguments/$DATASET/$SCENE.py"
        fi
        if [[ "$SKIP_EVAL" != "true" ]]; then
            echo "   CUDA_VISIBLE_DEVICES=$GPU python metrics.py --model_path $SAVE_PATH/$DATASET/${SCENE}_${DATASET}_${SCENE}"
        fi
    else
        # Setup environment
        echo "🔧 Setting up environment..."
        source bin/setup_wandb_team.sh
        
        echo "🚀 Starting training..."
        echo "📊 Training progress will be shown below:"
        echo "========================================"
        
        # Training with real-time output
        CUDA_VISIBLE_DEVICES=$GPU python train.py -s $GT_PATH/$SCENE --port 0 --model_path $SAVE_PATH/$DATASET/${SCENE}_${DATASET}_${SCENE} --expname "$EXP_NAME" --configs arguments/$DATASET/$SCENE.py -r $RESOLUTION
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ Training completed successfully!"
            
            if [[ "$SKIP_RENDER" != "true" ]]; then
                echo ""
                echo "🎨 Starting rendering..."
                echo "========================================"
                CUDA_VISIBLE_DEVICES=$GPU python render.py --model_path $SAVE_PATH/$DATASET/${SCENE}_${DATASET}_${SCENE} --skip_train --configs arguments/$DATASET/$SCENE.py
                
                if [ $? -eq 0 ]; then
                    echo "✅ Rendering completed successfully!"
                else
                    echo "❌ Rendering failed!"
                    exit 1
                fi
            fi
            
            if [[ "$SKIP_EVAL" != "true" ]]; then
                echo ""
                echo "📈 Starting evaluation..."
                echo "========================================"
                CUDA_VISIBLE_DEVICES=$GPU python metrics.py --model_path $SAVE_PATH/$DATASET/${SCENE}_${DATASET}_${SCENE}
                
                if [ $? -eq 0 ]; then
                    echo "✅ Evaluation completed successfully!"
                else
                    echo "❌ Evaluation failed!"
                    exit 1
                fi
            fi
            
            echo ""
            echo "🎉 Experiment completed successfully!"
            echo "🔗 Check your wandb dashboard: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
            echo "📁 Results saved to: $SAVE_PATH/$DATASET/${SCENE}_${DATASET}_${SCENE}"
        else
            echo "❌ Training failed!"
            exit 1
        fi
    fi
fi 