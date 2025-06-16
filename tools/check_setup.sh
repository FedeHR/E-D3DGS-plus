#!/bin/bash

# E-D3DGS Setup Verification Script
# Checks if everything is ready for experiments

echo "🔍 E-D3DGS Setup Verification"
echo "============================="

# Check if scripts are executable
echo "📋 Checking scripts..."
if [ -x "./bin/run_experiment.sh" ]; then
    echo "   ✅ bin/run_experiment.sh is executable"
else
    echo "   ❌ bin/run_experiment.sh is not executable"
    echo "      Fix: chmod +x bin/run_experiment.sh"
fi

if [ -x "./bin/run_batch_experiments.sh" ]; then
    echo "   ✅ bin/run_batch_experiments.sh is executable"
else
    echo "   ❌ bin/run_batch_experiments.sh is not executable"
    echo "      Fix: chmod +x bin/run_batch_experiments.sh"
fi

# Check configuration files
echo ""
echo "📁 Checking configuration files..."
if [ -d "experiments/configs" ]; then
    echo "   ✅ experiments/configs/ directory exists"
    config_count=$(ls experiments/configs/*.conf 2>/dev/null | wc -l)
    echo "   📊 Found $config_count configuration files:"
    ls experiments/configs/*.conf 2>/dev/null | sed 's/^/      /'
else
    echo "   ❌ experiments/configs/ directory missing"
fi

# Check required tools
echo ""
echo "🛠️  Checking required tools..."
if command -v bc &> /dev/null; then
    echo "   ✅ bc (calculator) is available"
else
    echo "   ❌ bc (calculator) is missing"
    echo "      Fix: sudo apt install bc"
fi

if command -v sbatch &> /dev/null; then
    echo "   ✅ SLURM (sbatch) is available"
else
    echo "   ⚠️  SLURM (sbatch) is not available (SLURM features disabled)"
fi

# Check Python environment
echo ""
echo "🐍 Checking Python environment..."
if [ "$CONDA_DEFAULT_ENV" = "ed3dgs" ]; then
    echo "   ✅ ed3dgs conda environment is active"
else
    echo "   ⚠️  ed3dgs conda environment is not active"
    echo "      Fix: conda activate ed3dgs"
fi

# Check if train.py exists and has required parameters
echo ""
echo "🚂 Checking training script..."
if [ -f "train.py" ]; then
    echo "   ✅ train.py exists"
    
    # Check if the script has the required parameters
    if grep -q "gaussian_embedding_dim" train.py; then
        echo "   ✅ gaussian_embedding_dim parameter is available"
    else
        echo "   ❌ gaussian_embedding_dim parameter is missing"
    fi
    
    if grep -q "temporal_embedding_dim" train.py; then
        echo "   ✅ temporal_embedding_dim parameter is available"
    else
        echo "   ❌ temporal_embedding_dim parameter is missing"
    fi
    
    if grep -q "use_fourier_features" train.py; then
        echo "   ✅ use_fourier_features parameter is available"
    else
        echo "   ❌ use_fourier_features parameter is missing"
    fi
else
    echo "   ❌ train.py is missing"
fi

# Check dataset structure
echo ""
echo "📂 Checking dataset structure..."
if [ -d "data" ]; then
    echo "   ✅ data/ directory exists"
    dataset_count=$(ls -1 data/ 2>/dev/null | wc -l)
    echo "   📊 Found $dataset_count datasets:"
    ls -1 data/ 2>/dev/null | head -5 | sed 's/^/      /'
    if [ $dataset_count -gt 5 ]; then
        echo "      ... and $(($dataset_count - 5)) more"
    fi
else
    echo "   ⚠️  data/ directory not found"
    echo "      Note: You'll need to specify --gt_path when running experiments"
fi

# Check wandb setup
echo ""
echo "🔗 Checking wandb setup..."
if [ -f "bin/setup_wandb_team.sh" ]; then
    echo "   ✅ bin/setup_wandb_team.sh exists"
    if grep -q "WANDB_PROJECT" bin/setup_wandb_team.sh; then
        project=$(grep "WANDB_PROJECT" bin/setup_wandb_team.sh | cut -d'"' -f2)
        echo "   📊 Wandb project: $project"
    fi
    if grep -q "WANDB_ENTITY" bin/setup_wandb_team.sh; then
        entity=$(grep "WANDB_ENTITY" bin/setup_wandb_team.sh | cut -d'"' -f2)
        echo "   👥 Wandb entity: $entity"
    fi
else
    echo "   ❌ bin/setup_wandb_team.sh is missing"
fi

# Test script functionality
echo ""
echo "🧪 Testing script functionality..."
echo "   Testing help command..."
if ./bin/run_experiment.sh --help > /dev/null 2>&1; then
    echo "   ✅ bin/run_experiment.sh --help works"
else
    echo "   ❌ bin/run_experiment.sh --help failed"
fi

echo "   Testing dry run..."
if ./bin/run_experiment.sh --dataset dynerf --scene cut_roasted_beef --no_slurm --dry_run > /dev/null 2>&1; then
    echo "   ✅ bin/run_experiment.sh --dry_run works"
else
    echo "   ❌ bin/run_experiment.sh --dry_run failed"
fi

# Summary
echo ""
echo "📋 Summary"
echo "=========="
echo "✅ Ready to use examples:"
echo ""
echo "   # Original defaults with SLURM (default mode)"
echo "   ./bin/run_experiment.sh --dataset dynerf --scene cut_roasted_beef"
echo ""
echo "   # Run locally without SLURM"
echo "   ./bin/run_experiment.sh --dataset dynerf --scene cut_roasted_beef --no_slurm"
echo ""
echo "   # Custom embedding dimensions"
echo "   ./bin/run_experiment.sh --dataset dynerf --scene cut_roasted_beef --gdim 8"
echo ""
echo "   # Enable Fourier features"
echo "   ./bin/run_experiment.sh --dataset dynerf --scene cut_roasted_beef --fourier_scale 4.0"
echo ""
echo "   # Batch experiments"
echo "   ./bin/run_batch_experiments.sh experiments/configs/default_batch.conf"
echo ""
echo "🎯 Everything is configured to use ORIGINAL DEFAULTS unless explicitly specified!"
echo ""
echo "📖 For more details, see: EXPERIMENT_SLURM_GUIDE.md" 