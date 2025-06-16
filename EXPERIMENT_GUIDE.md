# 🚀 E-D3DGS Experiment Guide

**E-D3DGS** (Enhanced Dynamic 3D Gaussian Splatting) extends static 3D Gaussian Splatting to handle temporal dynamics in scenes, with optional Fourier feature enhancements for improved spatial representation.

## 🎯 Research Focus

This project investigates:
- **Dynamic 3D Gaussian Splatting**: Temporal modeling for time-varying scenes
- **Fourier Feature Enhancement**: Frequency domain features for spatial detail
- **Embedding Optimization**: Gaussian and temporal embedding dimensions
- **Multi-dataset Evaluation**: DyNeRF, HyperNeRF, and Technicolor datasets

## 📋 Quick Start

### 1. Setup Environment
```bash
# Activate conda environment
conda activate ed3dgs

# Setup Wandb for experiment tracking
source bin/setup_wandb_team.sh

# Verify everything works
./tools/check_setup.sh
```

### 2. Run Your First Experiment
```bash
# Basic experiment (SLURM cluster)
./bin/run_experiment.sh --scene cut_roasted_beef

# Local execution (for debugging)
./bin/run_experiment.sh --scene cut_roasted_beef --no_slurm

# Monitor progress
./tools/monitor_experiments.sh --status
./tools/monitor_experiments.sh --watch
```

### 3. View Results
- **Wandb Dashboard**: https://wandb.ai/harjes-ludwig-maximilianuniversity-of-munich/E-D3DGS
- **Local Results**: `results/dynerf/cut_roasted_beef_*/`

## 🎮 Running Experiments

### Single Experiments

**Basic Usage:**
```bash
# Original implementation (baseline)
./bin/run_experiment.sh --scene cut_roasted_beef

# Custom embedding dimensions
./bin/run_experiment.sh --scene cut_roasted_beef --gdim 64 --tdim 512

# Enable Fourier features
./bin/run_experiment.sh --scene cut_roasted_beef --fourier_scale 2.0

# Full custom experiment
./bin/run_experiment.sh --scene vrig-chicken --gdim 16 --fourier_scale 4.0
```

**Execution Options:**
```bash
# SLURM cluster (default)
./bin/run_experiment.sh --scene cut_roasted_beef

# Local execution with real-time progress
./bin/run_experiment.sh --scene cut_roasted_beef --no_slurm

# High-priority partition
./bin/run_experiment.sh --scene cut_roasted_beef --abakus

# Dry run (see what would execute)
./bin/run_experiment.sh --scene cut_roasted_beef --dry_run
```

### Batch Experiments
```bash
# Run predefined experiment sets
./bin/run_batch_experiments.sh experiments/configs/default_batch.conf
./bin/run_batch_experiments.sh experiments/configs/fourier_comparison.conf
./bin/run_batch_experiments.sh experiments/configs/embedding_dimensions.conf
```

### Experiment Parameters

| Parameter | Default | Description | Research Impact |
|-----------|---------|-------------|-----------------|
| `--scene` | - | Scene name (auto-detects dataset) | **Required** - determines dataset and scene |
| `--gdim` | 32 | Gaussian embedding dimension | Model capacity and spatial detail |
| `--tdim` | 256 | Temporal embedding dimension | Temporal modeling capability |
| `--fourier_scale` | 0 | Fourier features scale (0=disabled) | Spatial frequency enhancement |
| `--embedding_init` | random | Embedding initialization (random/zero/xavier) | Training convergence |
| `--gpu` | 0 | GPU device ID | Hardware selection |
| `--resolution` | 2 | Resolution scaling factor | Training speed vs quality |

**Auto-detected Datasets:**
- **DyNeRF**: coffee_martini, cook_spinach, cut_roasted_beef, flame_salmon_1, flame_steak, sear_steak
- **HyperNeRF**: aleks-teapot, chickchicken, cut-lemon, hand, slice-banana, torchocolate, americano, cross-hands, espresso, keyboard, oven-mitts, split-cookie, tamping, 3dprinter, broom, vrig-chicken, peel-banana

## 📊 Enhanced Wandb Tracking & Metrics

### 🎯 Core Metrics (Every 100 Iterations)

**Most Important Metrics:**
- **`Number of Gaussians`** ⭐ - Number of 3D Gaussians in the model (tracks model growth)
- **`iterations`** - Current training iteration
- **`learning_rate`** - Current learning rate
- **`dataset`** - Auto-detected dataset type (dynerf/hypernerf)

### 📈 Training Metrics (Every 100 Iterations)

**Image Quality Metrics:**
- **`train/L1`** - L1 loss on training data (lower = better reconstruction)
- **`train/PSNR`** - Peak Signal-to-Noise Ratio in dB (higher = better quality)
- **`train/SSIM`** - Structural Similarity Index (higher = better perceptual quality)
- **`train/LPIPS`** - Learned Perceptual Image Patch Similarity (lower = better perceptual quality)

**System Metrics:**
- **`system/memory_allocated`** - GPU memory allocated in GB
- **`system/memory_reserved`** - GPU memory reserved in GB

### 🧪 Test Evaluation Metrics (Every 1000 Iterations)

**Comprehensive Test Evaluation:**
- **`test/L1`** - L1 loss on test cameras (generalization quality)
- **`test/PSNR`** - PSNR on test cameras (reconstruction quality)
- **`test/SSIM`** - SSIM on test cameras (structural similarity)
- **`test/LPIPS`** - LPIPS on test cameras (perceptual similarity)

*Test evaluation uses a random subset of up to 10 test cameras for efficiency*

### 📋 Experiment Metadata & State Tracking

**Experiment Lifecycle:**
- **`Name`** - Comprehensive run name: `{scene}_{user}_{hostname}_{timestamp}`
- **`Notes`** - Descriptive experiment notes with key parameters
- **`User`** - Username from environment
- **`Tags`** - [dataset_type, scene_name, user_tag, gdim_tag, tdim_tag, fourier_tag]
- **`Created`** - Automatic timestamp
- **`total_runtime_seconds`** - Total training time in seconds (logged at completion)

### 🔧 Comprehensive Configuration Logging

**Model Architecture Parameters:**
- `gaussian_embedding_dim`: Gaussian embedding dimension (affects spatial modeling capacity)
- `temporal_embedding_dim`: Temporal embedding dimension (affects temporal modeling capacity)
- `use_fourier_features`: Whether Fourier features are enabled (boolean)
- `fourier_scale`: Fourier feature scale factor (0 = disabled, higher = more detail)
- `sh_degree`: Spherical harmonics degree for color representation

**Training Configuration:**
- `iterations`: Total training iterations (affects training duration)
- `batch_size`: Batch size for training (affects memory usage and convergence)
- `resolution`: Resolution scaling factor (affects image quality vs speed)

**Learning Rate Schedule:**
- `position_lr_init`: Initial learning rate for Gaussian positions
- `position_lr_final`: Final learning rate for Gaussian positions
- `deformation_lr_init`: Initial learning rate for temporal deformations
- `deformation_lr_final`: Final learning rate for temporal deformations
- `feature_lr`: Learning rate for color features
- `opacity_lr`: Learning rate for opacity values
- `scaling_lr`: Learning rate for Gaussian scaling
- `rotation_lr`: Learning rate for Gaussian rotation

**Loss Weights & Regularization:**
- `lambda_dssim`: Weight for DSSIM loss component (perceptual loss)
- `reg_coef`: Regularization coefficient for embedding smoothness
- `opacity_l1_coef_fine`: L1 regularization on opacity values
- `coef_tv_temporal_embedding`: Total variation regularization on temporal embeddings

**Densification Parameters:**
- `densify_from_iter`: Iteration to start densification
- `densify_until_iter`: Iteration to stop densification
- `densification_interval`: Interval between densification steps
- `opacity_reset_interval`: Interval for opacity reset
- `densify_grad_threshold`: Gradient threshold for densification

**System & Reproducibility:**
- `hostname`: Compute node name (for cluster tracking)
- `username`: User running the experiment
- `git_commit`: Git commit hash (first 8 characters for reproducibility)

### 📊 What These Metrics Tell You

**Training Progress:**
- **Number of Gaussians**: Shows model growth during densification phases (typically 50k-200k+)
- **train/L1**: Overall reconstruction error - should decrease over time
- **train/PSNR**: Image quality in dB - higher values (>25 dB) indicate good reconstruction
- **train/SSIM**: Structural similarity - values closer to 1.0 are better

**Generalization Quality:**
- **test/PSNR**: How well the model generalizes to unseen viewpoints
- **test/SSIM**: Perceptual quality on test views
- **test/LPIPS**: Learned perceptual similarity - lower values indicate better perceptual quality

**System Health:**
- **Memory metrics**: Track GPU usage to avoid OOM errors
- **Learning rate**: Shows training schedule progression

### Experiment Naming Convention

Experiments are automatically named to reflect their configuration:
```
{scene}_{user}_{hostname}_{timestamp}

Examples:
- cut_roasted_beef_kerschern_buchit_20250616_023306
- vrig-chicken_kerschern_cancalit_20250616_143022
```

**Tags for Organization:**
```
[dynerf, cut_roasted_beef, user_kerschern, gdim_32, tdim_256]
[hypernerf, vrig-chicken, user_kerschern, gdim_16, tdim_128, fourier_2.0]
```

## 📈 Monitoring Experiments

### Monitoring Commands

| Command | Description | Example |
|---------|-------------|---------|
| `--status, -s` | Show all experiment statuses | `./tools/monitor_experiments.sh -s` |
| `--logs, -l [JOB]` | Show logs for specific job | `./tools/monitor_experiments.sh -l 12369` |
| `--watch, -w [JOB]` | Watch logs in real-time | `./tools/monitor_experiments.sh -w` |

### File Organization

**Automatic Organization by Dataset:**
```
experiments/
├── slurm_jobs/           # SLURM scripts organized by dataset
│   ├── dynerf/          # DyNeRF experiments
│   └── hypernerf/       # HyperNeRF experiments
└── slurm_logs/          # Logs organized by dataset
    ├── dynerf/
    └── hypernerf/
```

**Smart Naming (Only Non-Default Parameters):**
```
dynerf_cut_roasted_beef_20250616_021146.sh                    # Default parameters
dynerf_cut_roasted_beef_gdim64_20250616_021146.sh            # Custom Gaussian dim
dynerf_cut_roasted_beef_fourier2_20250616_021146.sh          # Fourier features
hypernerf_vrig-chicken_gdim16_fourier4_20250616_021146.sh    # Full custom
```

## 🗂️ Project Structure

```
E-D3DGS-plus/
├── bin/                    # 🔧 Main executable scripts
│   ├── run_experiment.sh   # Single experiment runner
│   ├── run_batch_experiments.sh # Batch runner
│   └── setup_wandb_team.sh # Wandb configuration
├── tools/                  # 🛠️ Utility scripts
│   ├── monitor_experiments.sh # Experiment monitoring
│   └── check_setup.sh      # Setup verification
├── experiments/            # 📊 Experiment management
│   ├── configs/           # Batch experiment configs
│   ├── slurm_jobs/       # Generated job scripts (by dataset)
│   └── slurm_logs/       # Job execution logs (by dataset)
├── data/                  # 📂 Datasets
├── results/               # 📈 Experiment outputs
├── train.py              # Enhanced training script with comprehensive logging
├── render.py             # Rendering script
└── metrics.py            # Evaluation script
```

## 🔧 Advanced Configuration

### Custom Batch Experiments

Create your own batch configuration:
```bash
# experiments/configs/my_experiment.conf
# Each line represents one experiment with its parameters
--scene cut_roasted_beef
--scene cut_roasted_beef --gdim 64
--scene cut_roasted_beef --fourier_scale 2.0
--scene vrig-chicken --gdim 16 --fourier_scale 4.0
```

### SLURM Configuration

**Available Partitions:**
- **NvidiaAll** (default): NVIDIA GPU partition for general use
- **Abakus**: High-priority partition (use `--abakus` flag)

**Automatic Configuration:**
- Memory specifications removed to avoid SLURM errors
- Default to NvidiaAll partition (most reliable)
- Auto-detection of dataset types
- Organized file structure by dataset

## 🎯 Research Workflows

### Baseline Comparison
```bash
# Run baseline experiments
./bin/run_batch_experiments.sh experiments/configs/default_batch.conf
```

### Fourier Feature Study
```bash
# Compare with/without Fourier features
./bin/run_batch_experiments.sh experiments/configs/fourier_comparison.conf
```

### Embedding Dimension Analysis
```bash
# Test different embedding sizes
./bin/run_batch_experiments.sh experiments/configs/embedding_dimensions.conf
```

### Custom Research Question
```bash
# Single targeted experiment
./bin/run_experiment.sh --scene vrig-chicken --gdim 16 --fourier_scale 2.0
```

## 🚨 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Job pending | Wait for resources or try `--abakus` for high priority |
| Conda environment not found | Run `conda activate ed3dgs` |
| Wandb not logging | Check `source bin/setup_wandb_team.sh` |
| Permission denied | Check file permissions with `chmod +x bin/run_experiment.sh` |
| Out of memory | Reduce `--resolution` or use smaller `--gdim`/`--tdim` |
| Training stuck at "Loading image" | Normal for first iteration - wait for training to start |

### Getting Help

1. **Check setup**: `./tools/check_setup.sh`
2. **Test dry run**: `./bin/run_experiment.sh --scene cut_roasted_beef --dry_run`
3. **View logs**: `./tools/monitor_experiments.sh --logs JOB_ID`
4. **Monitor wandb**: https://wandb.ai/harjes-ludwig-maximilianuniversity-of-munich/E-D3DGS

## 🎯 Quick Reference

### Most Common Commands

```bash
# === RUNNING EXPERIMENTS ===
# Basic experiment (original defaults)
./bin/run_experiment.sh --scene cut_roasted_beef

# Custom experiment with Fourier features
./bin/run_experiment.sh --scene vrig-chicken --gdim 16 --fourier_scale 2.0

# Local execution for debugging
./bin/run_experiment.sh --scene cut_roasted_beef --no_slurm

# === MONITORING EXPERIMENTS ===
# Check all experiment statuses
./tools/monitor_experiments.sh --status

# Watch latest experiment in real-time
./tools/monitor_experiments.sh --watch

# View logs for specific job
./tools/monitor_experiments.sh --logs 12369

# === SLURM COMMANDS ===
# Check job status
squeue -u $USER

# Cancel a job
scancel JOB_ID
```

### Key Features

**✅ What This System Provides:**
- 🎯 **Auto-detection** of dataset types from scene names
- 📁 **Organized file structure** by dataset type
- 📝 **Smart naming** showing only non-default parameters
- 📊 **Comprehensive wandb logging** with 20+ metrics including Number of Gaussians
- 🔍 **Easy monitoring** with built-in tools
- 🖥️ **Both SLURM and local execution** options
- 🧹 **Clean, minimal codebase** focused on essential functionality
- 📈 **Enhanced test evaluation** every 1000 iterations
- 🎛️ **Complete parameter tracking** for reproducibility

**🎉 Ready to run E-D3DGS experiments with comprehensive tracking!** Start with the Quick Start section and monitor your experiments on Wandb! 🚀 