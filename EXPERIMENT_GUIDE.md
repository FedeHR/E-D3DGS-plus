# 🚀 E-D3DGS PLUS Experiment Guide

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
# Activate conda environment (REQUIRED!)
conda activate ed3dgs

# Setup Wandb for experiment tracking
source bin/setup_wandb_team.sh

# Verify everything works
./tools/check_setup.sh
```

**⚠️ Important**: The conda environment **must be activated** before running experiments. The pre-flight checks require packages like `psutil` that are only available in the `ed3dgs` environment. If you see import errors, ensure conda is activated.

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
- **Local Results**: `results/`
- **Local Wandb Data**: `wandb/` (detailed experiment tracking - see below)

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



### Experiment Parameters

| Parameter | Default | Description | Research Impact |
|-----------|---------|-------------|-----------------|
| `--scene` | - | Scene name (auto-detects dataset) | **Required** - determines dataset and scene |
| `--gdim` | 32 | Gaussian embedding dimension | Model capacity and spatial detail |
| `--tdim` | 256 | Temporal embedding dimension | Temporal modeling capability |
| `--fourier_scale` | 0 | Fourier features scale (0=disabled) | Spatial frequency enhancement |
| `--embedding_init` | zero | Gaussian embedding initialization (see methods below) | Training convergence & spatial detail |
| `--temporal_embedding_init` | normal | Temporal embedding initialization | Temporal modeling stability |
| `--num_freq_bands` | auto | Frequency bands for structured Fourier | Frequency coverage range |
| `--gpu` | 0 | GPU device ID | Hardware selection |
| `--resolution` | 2 | Resolution scaling factor | Training speed vs quality |
| `--time` | auto | Custom time limit (HH:MM:SS format) | SLURM job duration control |

**Auto-detected Datasets:**
- **DyNeRF**: coffee_martini, cook_spinach, cut_roasted_beef, flame_salmon_1, flame_steak, sear_steak
- **HyperNeRF**: aleks-teapot, chickchicken, cut-lemon, hand, slice-banana, torchocolate, americano, cross-hands, espresso, keyboard, oven-mitts, split-cookie, tamping, 3dprinter, broom, vrig-chicken, peel-banana

## 🧬 Comprehensive Embedding Initialization Methods

Based on **"Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"** (Tancik et al., NeurIPS 2020) and classical deep learning initialization strategies:

### 🎯 Classical Methods
- **`zero`** (default): Zero initialization - stable baseline, good for ablations
- **`normal`**: Standard normal distribution N(0,0.01²) - basic random initialization  
- **`random`**: Alias for normal initialization
- **`uniform`**: Uniform distribution U(-0.01,0.01) - bounded random initialization

### ⚖️ Xavier/Glorot Family (Better Gradient Flow)
- **`xavier`**: Xavier normal initialization - recommended for linear/tanh activations
- **`xavier_normal`**: Same as xavier (explicit naming)
- **`xavier_uniform`**: Xavier uniform variant - sometimes more stable

### 🎢 Kaiming/He Family (ReLU Networks)
- **`kaiming`**: Kaiming normal initialization - optimal for ReLU networks
- **`kaiming_normal`**: Same as kaiming (explicit naming) 
- **`he_normal`**: Same as kaiming_normal (alternative naming)
- **`he_uniform`**: He uniform variant - for ReLU activations

### 🌊 Fourier & Positional Encoding Methods (High-Frequency Spatial Details)

These methods enable neural networks to learn high-frequency functions by mapping coordinates to higher-dimensional feature spaces. Based on **"Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"** (Tancik et al., NeurIPS 2020).

#### 🎯 Random Fourier Features (`fourier`, `positional`)
**What it does**: Maps 3D coordinates (x,y,z) through random frequency projections
**Formula**: `γ(p) = [cos(2πB₁·p), sin(2πB₁·p), ..., cos(2πBₘ·p), sin(2πBₘ·p)]`

**Parameters:**
- `--fourier_scale`: Controls frequency range [0, scale] (higher = more detail)
- No `--num_freq_bands` needed (auto-calculated from dimension)

**Dimension mapping:**
- **16D**: 8 random frequency vectors → broad frequency coverage
- **32D**: 16 random frequency vectors → denser frequency sampling  
- **64D**: 32 random frequency vectors → very dense sampling

**Best for**: General scenes where you want better spatial detail than normal embeddings

#### 🏗️ Structured Fourier / NeRF Positional Encoding (`structured_fourier`, `positional_encoding`)
**What it does**: Uses deterministic powers-of-2 frequencies like NeRF
**Formula**: `γ(p) = [p, sin(2⁰πp), cos(2⁰πp), sin(2¹πp), cos(2¹πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp)]`

**Parameters:**
- `--fourier_scale`: Base frequency (default: 1.0)
- `--num_freq_bands`: Number of frequency bands (auto-calculated if not specified)

**Dimension mapping & frequency ranges:**
- **16D**: ~2 freq bands → [scale, 2×scale] (e.g., [1.0, 2.0])
- **32D**: ~5 freq bands → [scale, 16×scale] (e.g., [1.0, 16.0])
- **64D**: ~10 freq bands → [scale, 512×scale] (e.g., [1.0, 512.0])

**Best for**: Scenes with predictable frequency content, systematic ablations

#### 🧠 Learned Fourier Features (`learned_fourier`)
**What it does**: Initializes with structured random frequencies but allows learning
**Features**: Multi-scale initialization (1/3 low, 1/3 medium, 1/3 high frequencies)

**Parameters:**
- `--fourier_scale`: Controls initial frequency distribution
- No `--num_freq_bands` needed

**Best for**: Complex scenes where optimal frequencies are unknown

### 🕒 Temporal-Specific Methods
- **`sinusoidal`**: Transformer-style positional encoding for time (temporal embeddings only)

### 📊 **CRITICAL: Fourier vs Normal Embedding Dimension Trade-offs**

This is key for understanding how to choose parameters:

#### 🔍 **Frequency Coverage vs Embedding Dimension**

**Random Fourier (`fourier`) vs Normal Embeddings:**
```bash
# These provide roughly equivalent model capacity:
--embedding_init fourier --fourier_scale 2.0 --gdim 16  # 8 freq mappings
--embedding_init xavier --gdim 16                        # 16 normal dimensions

# But Fourier has better high-frequency representation!
```

**Structured Fourier (`structured_fourier`) vs Normal:**
```bash
# Low-frequency focused (good for smooth scenes):
--embedding_init structured_fourier --fourier_scale 1.0 --num_freq_bands 3 --gdim 32
# Covers frequencies [1.0, 2.0, 4.0] = 3×6+3 = 21 effective dims

# High-frequency focused (good for detailed scenes):  
--embedding_init structured_fourier --fourier_scale 1.0 --num_freq_bands 8 --gdim 64
# Covers frequencies [1.0, 2.0, 4.0, ..., 128.0] = 8×6+3 = 51 effective dims

# Equivalent normal embedding:
--embedding_init xavier --gdim 64  # 64 normal dimensions (less structured)
```

#### ⚖️ **How to Choose: Fourier Bands vs Higher Dimensions**

**For the SAME computational cost, you can choose:**

**Option A: More Fourier Frequency Bands (Structured)**
```bash
--embedding_init structured_fourier --fourier_scale 1.0 --num_freq_bands 10 --gdim 64
```
- **Pros**: Explicit frequency control, interpretable, good for known detail levels
- **Cons**: Fixed frequency structure, may miss optimal frequencies

**Option B: Higher Normal Embedding Dimension**
```bash
--embedding_init xavier --gdim 64
```
- **Pros**: Maximum flexibility, can learn any frequency, adaptive
- **Cons**: Less interpretable, may struggle with high frequencies initially

**Option C: Random Fourier with Higher Scale**
```bash
--embedding_init fourier --fourier_scale 4.0 --gdim 64
```
- **Pros**: Random frequency sampling, good high-freq initialization, flexible
- **Cons**: Non-deterministic, less controlled than structured

#### 🎯 **Practical Decision Guidelines**

**Use Higher `num_freq_bands` when:**
- You know the scene has specific detail scales
- You want interpretable frequency control
- You're doing systematic frequency ablations
- Scene has obvious geometric detail levels

**Use Higher `gdim` (normal embeddings) when:**
- Scene complexity is unknown
- You want maximum model flexibility  
- You prefer simple, well-tested initialization
- Computational cost is not a constraint

**Use Random Fourier (`fourier`) when:**
- You want better high-frequency capability than normal
- You want some frequency benefits without manual tuning
- You're prototyping or want good general performance

#### 📐 **Memory & Speed Considerations**

All methods have the same computational cost for the same `gdim`, but:
- **Fourier methods**: Better gradient flow for high-frequency details
- **Normal embeddings**: Slightly faster initialization
- **Structured Fourier**: Most interpretable results

### 🔬 Recommended Configurations & Comparisons

#### 🎯 **High-Detail Scenes** (cook_spinach, cut_roasted_beef, flame_salmon)
```bash
# Option A: Random Fourier (balanced performance)
--embedding_init fourier --fourier_scale 2.0 --gdim 32

# Option B: Structured Fourier (interpretable frequencies)  
--embedding_init structured_fourier --fourier_scale 1.0 --num_freq_bands 8 --gdim 64

# Option C: High-dimension normal (maximum flexibility)
--embedding_init xavier --gdim 64
```

#### ⚡ **Fast Prototyping & Development**
```bash
# Minimal setup (fastest training)
--embedding_init xavier --gdim 16 --tdim 128

# Balanced setup (good quality/speed)
--embedding_init kaiming --gdim 32 --tdim 256

# Quick Fourier test (better detail than normal)
--embedding_init fourier --fourier_scale 1.0 --gdim 32
```

#### 🕒 **Temporal Stability & Motion Modeling**
```bash
# Smooth temporal transitions
--temporal_embedding_init sinusoidal --tdim 256

# Better gradient flow for temporal features
--temporal_embedding_init xavier --tdim 512

# Large temporal capacity for complex motion
--temporal_embedding_init xavier --tdim 1024
```

#### 🔬 **Scientific Experiments & Ablations**

**Initialization Method Comparison:**
```bash
# Baseline (standard approach)
--embedding_init zero --gdim 32

# Better gradient flow
--embedding_init xavier --gdim 32   

# High-frequency capability
--embedding_init fourier --fourier_scale 2.0 --gdim 32

# Structured frequency analysis
--embedding_init structured_fourier --fourier_scale 1.0 --num_freq_bands 5 --gdim 32
```

**Frequency Band Analysis (structured_fourier):**
```bash
# Low frequency focus (smooth scenes)
--embedding_init structured_fourier --num_freq_bands 3 --gdim 32

# Medium frequency focus (balanced)
--embedding_init structured_fourier --num_freq_bands 6 --gdim 32  

# High frequency focus (detailed scenes)
--embedding_init structured_fourier --num_freq_bands 10 --gdim 64
```

**Equivalent Capacity Comparisons:**
```bash
# ~16 effective dimensions - test which works better:
--embedding_init xavier --gdim 16                         # Normal baseline
--embedding_init fourier --fourier_scale 1.0 --gdim 16   # Random Fourier
--embedding_init structured_fourier --num_freq_bands 2 --gdim 16  # Structured

# ~32 effective dimensions:  
--embedding_init xavier --gdim 32                         # Normal baseline
--embedding_init fourier --fourier_scale 2.0 --gdim 32   # Random Fourier
--embedding_init structured_fourier --num_freq_bands 5 --gdim 32  # Structured
```

#### 🎮 **Quick Scene-Specific Recommendations**

**Smooth Scenes** (vrig-chicken, hand):
```bash
--embedding_init xavier --gdim 32 --temporal_embedding_init sinusoidal
```

**Detailed/Textured Scenes** (cook_spinach, cut_roasted_beef):
```bash
--embedding_init fourier --fourier_scale 2.0 --gdim 64
```

**Fast-Moving Scenes** (flame_salmon, sear_steak):
```bash
--embedding_init structured_fourier --num_freq_bands 8 --gdim 64 --temporal_embedding_init xavier --tdim 512
```

## 📊 Enhanced Wandb Tracking & Metrics

### 🖼️ **NEW: Visual Progress Tracking**

**Real vs Generated Image Comparison:**
- **Frequency**: Every 1000 iterations during test evaluation
- **Location**: Wandb **Media** section with iteration slider
- **Content**: Side-by-side comparison of real ground truth vs generated images
- **Navigation**: Use the step slider to see training progress visually
- **Camera**: Uses first test camera for consistency

This creates the exact interface you see in Wandb with "Real Image" and "Generated Image" that you can scrub through with the iteration slider! 🎬

**Example Wandb Media View:**
```
Media                                          1-2 of 2
┌─────────────────────┐
│   Real Image        │  ← Ground truth
│   [kitchen scene]   │
└─────────────────────┘
Step [======●=========] 80000

┌─────────────────────┐ 
│ Generated Image     │  ← Model output
│ [kitchen scene]     │
└─────────────────────┘
Step [======●=========] 80000
```

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

## 📈 Monitoring & Logging

### Enhanced Experiment Monitoring

**New Features:**
- 🗂️ **Organized Log Structure**: Logs organized by experiment name
- 💾 **Memory Estimation**: Automatic memory requirement calculation
- 🔍 **Enhanced Crash Detection**: Detailed error reporting with system state
- 📊 **Real-time Memory Monitoring**: GPU memory usage tracked during training
- 🆔 **Job ID Integration**: Easy matching between SLURM jobs and log files

### Monitoring Commands

| Command | Description | Example |
|---------|-------------|---------|
| `--status, -s` | Show all experiment statuses | `./tools/monitor_experiments.sh -s` |
| `--logs, -l [JOB]` | Show logs for specific job | `./tools/monitor_experiments.sh -l 12369` |
| `--watch, -w [JOB]` | Watch logs in real-time | `./tools/monitor_experiments.sh -w` |

### Organized Log Structure

**📅 Chronologically Organized Structure:**
```
experiments/slurm_logs/
├── dynerf/                      # 🎯 DyNeRF experiments (chronologically sorted)
│   ├── 20250618_150350_cut_roasted_beef-gdim32-tdim256-structured_fourier4/
│   │   ├── 20250618_150350_12345.out        # Job output with timestamp_jobID
│   │   ├── 20250618_150350_12345.err        # Job errors with timestamp_jobID
│   │   ├── 20250618_150350_12345.progress   # Progress tracking
│   │   └── 20250618_150350_12345.monitor    # Memory monitoring
│   ├── 20250618_150400_cut_roasted_beef-gdim16-tdim256-xavier-temporalsinusoidal/
│   │   └── 20250618_150400_12346.*
│   └── 20250618_150500_cook_spinach-gdim64-tdim512-fourier2.0/
│       └── 20250618_150500_12347.*
├── hypernerf/                   # 🎯 HyperNeRF experiments (chronologically sorted)
│   ├── 20250618_150400_vrig-chicken-gdim64-tdim256-fourier2.0/
│   │   └── 20250618_150400_12348.*
│   └── 20250618_150600_hand-gdim32-tdim256-xavier/
│       └── 20250618_150600_12349.*
└── archive/                     # 🗄️ Old logs from before timestamp naming
    ├── dynerf/                  # Previous unorganized logs
    └── hypernerf/
```

**📂 Naming Convention:**
- **Directory**: `YYYYMMDD_HHMMSS_scene-gdim32-tdim256-method`  
- **Files**: `YYYYMMDD_HHMMSS_jobID.{out,err,progress,monitor}`
- **SLURM scripts**: `YYYYMMDD_HHMMSS_dataset_scene_method.sh`

### Enhanced Crash Detection

**Memory Issue Detection:**
```bash
=== JOB TERMINATED ===
Signal: SIGTERM
Time: 2025-06-17 16:30:45
Job ID: 12895
Likely cause: Memory limit exceeded (OOM)
=== SYSTEM INFO ===
GPU Memory: 15.2GB / 16GB (94% used)
System Memory: 63.2GB / 64GB (95% used)
==================
```

**Real-time Memory Monitoring:**
- GPU memory usage logged every 30 seconds
- Helps identify memory growth patterns
- Stored in `*.monitor` files with job ID

### Memory Management

**Automatic Memory Estimation:**
- **DyNeRF scenes**: Base 8GB + embedding scaling
- **HyperNeRF scenes**: Base 8GB + 1.5x embedding scaling  
- **Safety margin**: 30% additional allocation
- **Example**: gdim=32, tdim=256 → ~23GB estimated

**SLURM Integration:**
- Automatic `#SBATCH --mem=XG` allocation
- Prevents OOM crashes from insufficient memory requests
- Override with `--slurm_mem` if needed

## 🗂️ Project Structure

```
E-D3DGS-plus/
├── bin/                    # 🔧 Main executable scripts
│   ├── run_experiment.sh   # Single experiment runner
│   └── setup_wandb_team.sh # Wandb configuration
├── tools/                  # 🛠️ Utility scripts
│   ├── monitor_experiments.sh # Experiment monitoring
│   └── check_setup.sh      # Setup verification
├── experiments/            # 📊 Experiment management
│   ├── slurm_jobs/       # Generated job scripts (by dataset)
│   └── slurm_logs/       # Job execution logs (by dataset)
├── utils/                  # 🧬 Core utilities
│   └── simple_embedding_init.py # Comprehensive embedding initialization
├── data/                  # 📂 Datasets
├── results/               # 📈 Experiment outputs
├── train.py              # Enhanced training script with comprehensive logging
├── render.py             # Rendering script
└── metrics.py            # Evaluation script
```

## 🔧 Advanced Configuration

### Custom Experiment Series

Run systematic experiments with shell loops:
```bash
# Test different embedding dimensions
for gdim in 16 32 64; do
    ./bin/run_experiment.sh --scene cut_roasted_beef --gdim $gdim
done

# Test different Fourier scales
for scale in 1.0 2.0 4.0; do
    ./bin/run_experiment.sh --scene cut_roasted_beef --embedding_init fourier --fourier_scale $scale
done

# Test frequency bands for structured Fourier
for bands in 3 6 9; do
    ./bin/run_experiment.sh --scene cook_spinach --embedding_init structured_fourier --num_freq_bands $bands --gdim 64
done
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

### Systematic Comparisons
```bash
# Baseline comparison across scenes
./bin/run_experiment.sh --scene cut_roasted_beef --embedding_init zero
./bin/run_experiment.sh --scene vrig-chicken --embedding_init zero

# Fourier feature study  
./bin/run_experiment.sh --scene cut_roasted_beef --embedding_init fourier --fourier_scale 1.0
./bin/run_experiment.sh --scene cut_roasted_beef --embedding_init fourier --fourier_scale 2.0
./bin/run_experiment.sh --scene cut_roasted_beef --embedding_init fourier --fourier_scale 4.0

# Embedding dimension analysis
./bin/run_experiment.sh --scene cut_roasted_beef --gdim 16
./bin/run_experiment.sh --scene cut_roasted_beef --gdim 32  
./bin/run_experiment.sh --scene cut_roasted_beef --gdim 64
```

### Custom Research Question
```bash
# Single targeted experiment
./bin/run_experiment.sh --scene vrig-chicken --gdim 16 --fourier_scale 2.0
```

## 🖥️ SLURM Cluster Information & Commands

### 📊 Available Partitions & Nodes

| Partition | Nodes | Time Limit | QoS Required | Best For |
|-----------|-------|------------|--------------|----------|
| **NvidiaAll** | 25 NVIDIA GPU nodes | Unlimited | None | Standard GPU training |
| **Abaki** | 4 high-priority nodes (abakus11-12, 21-22) | 5 days | `abaki` | High-priority jobs |
| **AMD** | 66 AMD GPU nodes | Unlimited | None | AMD GPU training |
| **Krater** | 40 CPU nodes | Unlimited | None | CPU-only jobs |
| **All** | 131 mixed nodes | Unlimited | None | General purpose |

### 🎯 Abakus Partition (High-Priority)

**Key Features:**
- **4 nodes**: abakus11, abakus12, abakus21, abakus22
- **Default time limit**: 24 hours (we override to 4 hours)
- **Max time limit**: 5 days  
- **QoS required**: `abaki`
- **Reserved times**: Sunday & Monday 8:00-22:00 for compvis25 group
- **No memory specification**: Nodes don't support `--mem` parameter

**Custom Time Limits:**
- **Default**: `--abakus` sets 4:00:00 (4 hours)
- **Custom**: Use `--time HH:MM:SS` for specific durations
- **Examples**: 
  - `--time 7:30:00` (7 hours 30 minutes)
  - `--time 2:15:00` (2 hours 15 minutes)
  - `--time 12:00:00` (12 hours)
- **⚠️ Note**: Custom time overrides partition defaults

**Reservations (Auto-detected by script):**
- **Sunday**: `compvis25_So` (8:00-22:00)
- **Monday**: `compvis25_Mo` (8:00-22:00)

### 🔧 Essential SLURM Commands

#### Job Management
```bash
# Submit a job
sbatch my_script.sh

# Check all your jobs
squeue -u $USER

# Check specific partition
squeue -p Abaki
squeue -p NvidiaAll

# Check all jobs with detailed format
squeue -o "%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R"

# Get detailed job information
scontrol show job JOB_ID

# Cancel a job
scancel JOB_ID

# Cancel all your jobs
scancel -u $USER
```

#### Node Information
```bash
# Show all partitions
scontrol show partition

# Show specific partition details
scontrol show partition Abaki
scontrol show partition NvidiaAll

# Show node details
scontrol show node abakus12
scontrol show nodes

# Check node availability
sinfo
sinfo -p Abaki
```

#### Queue Analysis
```bash
# Show queue with reasons for pending jobs
squeue -l

# Show only running jobs
squeue -t RUNNING

# Show only pending jobs  
squeue -t PENDING

# Show jobs on specific nodes
squeue -w abakus11,abakus12

# Show reservation information
scontrol show reservation
```

#### Job History & Stats
```bash
# Show your recent jobs (last 24h)
sacct -S $(date -d '1 day ago' +%Y-%m-%d) -u $USER

# Show detailed job accounting
sacct -j JOB_ID --format=JobID,JobName,Partition,Account,AllocCPUS,State,ExitCode,Start,End,Elapsed,MaxRSS

# Show usage by partition
sreport cluster utilization start=2025-06-01 end=2025-06-30
```

### 🎯 Best Practices for Abakus

**For 4-Hour Jobs (Recommended):**
```bash
# Perfect for quick experiments
./bin/run_experiment.sh --scene cut_roasted_beef --abakus --gdim 16 --tdim 128

# For detailed scenes, use efficient settings
./bin/run_experiment.sh --scene vrig-chicken --abakus --gdim 32 --tdim 256
```

**For Longer Jobs (Use custom time or reservations):**
```bash
# Custom time limits (works with any partition)
./bin/run_experiment.sh --scene cut_roasted_beef --abakus --time 7:30:00

# Use Sunday/Monday reservations for longer runs
./bin/run_experiment.sh --scene cut_roasted_beef --abakus --reservation compvis25_So

# Combine custom time with reservations
./bin/run_experiment.sh --scene cut_roasted_beef --abakus --time 8:00:00 --reservation compvis25_So
```

**Resource Guidelines:**
- **Memory**: Don't specify `--mem` (automatically handled)
- **Time**: Default 4 hours (perfect for most experiments)
- **CPUs**: 8 CPUs per node (automatically allocated)

### 📈 Monitoring Your Jobs

**Real-time monitoring:**
```bash
# Watch your jobs continuously
watch -n 5 "squeue -u $USER"

# Monitor specific partition
watch -n 10 "squeue -p Abaki -o '%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R'"

# Check job progress
tail -f experiments/slurm_logs/dynerf/*/20250618_*.out
```

**Our Enhanced Monitoring:**
```bash
# Use our built-in tools
./tools/monitor_experiments.sh --status
./tools/monitor_experiments.sh --watch
./tools/monitor_experiments.sh --logs JOB_ID
```

**📅 Finding Your Experiments (Chronological Order):**
```bash
# List experiments by date (newest first)
ls -t experiments/slurm_logs/dynerf/
ls -t experiments/slurm_logs/hypernerf/

# Find today's experiments
ls experiments/slurm_logs/dynerf/ | grep $(date +%Y%m%d)

# Find experiments with specific method
ls experiments/slurm_logs/dynerf/ | grep fourier
ls experiments/slurm_logs/dynerf/ | grep structured_fourier

# Quick view of recent experiment names
ls -t experiments/slurm_logs/dynerf/ | head -5 | sed 's/^[0-9_]*_//'

# View logs from most recent experiment
tail -f experiments/slurm_logs/dynerf/$(ls -t experiments/slurm_logs/dynerf/ | head -1)/*.out
```

### ⚡ Quick Reference Commands

```bash
# === MOST USEFUL COMMANDS ===
# Check your current jobs
squeue -u $USER

# Check Abakus availability  
squeue -p Abaki

# Submit to Abakus with 4h limit
./bin/run_experiment.sh --scene SCENE --abakus

# Cancel a hanging job
scancel JOB_ID

# Check why job is pending
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R"

# Show detailed job info
scontrol show job JOB_ID

# Check node status
sinfo -p Abaki

# View reservations
scontrol show reservation | grep compvis25

# === LOG ORGANIZATION ===
# Organize old logs (run once)
./tools/organize_old_logs.sh

# View experiments chronologically
ls -t experiments/slurm_logs/dynerf/
ls -t experiments/slurm_logs/hypernerf/

# Find today's experiments
ls experiments/slurm_logs/dynerf/ | grep $(date +%Y%m%d)
```

## 📊 Local Wandb Experiment Tracking

### 🗂️ Wandb Folder Structure

The `wandb/` directory contains detailed tracking data for **every experiment run** locally. Each experiment creates a timestamped run directory with comprehensive logging:

```
wandb/
├── run-20250702_173239-ffel0omj/     ← Individual experiment run
│   ├── files/
│   │   ├── wandb-summary.json        ← Final metrics and results
│   │   ├── config.yaml               ← Complete experiment configuration
│   │   ├── wandb-metadata.json       ← System info and runtime details
│   │   ├── requirements.txt          ← Python dependencies snapshot
│   │   ├── conda-environment.yaml    ← Complete conda environment
│   │   ├── diff.patch               ← Code changes from last commit
│   │   └── code/
│   │       └── train.py             ← Exact code used for experiment
│   ├── logs/
│   │   ├── debug.log               ← Wandb internal logs
│   │   └── debug-internal.log      ← Detailed debugging info
│   └── tmp/                        ← Temporary files during run
├── latest-run/                     ← Symlink to most recent run
├── debug.log                       ← Global wandb debug log
└── debug-internal.log              ← Global internal logs
```

### 📋 What's Tracked in Each Run

#### **1. Final Results (`wandb-summary.json`)**
Key metrics from completed experiments:
```json
{
  "Number of Gaussians": 90573,
  "iterations": 80000,
  "test/PSNR": 27.17,
  "test/SSIM": 0.899,
  "test/LPIPS": 0.176,
  "train/PSNR": 27.30,
  "memory_allocated_GB": 1.03,
  "_runtime": 369.32
}
```

#### **2. Complete Configuration (`config.yaml`)**
Every experiment parameter is logged:
```yaml
experiment_name: "dynerf/cut_roasted_beef-gdim32-tdim256"
dataset_type: "dynerf"
scene_name: "cut_roasted_beef"
gaussian_embedding_dim: 32
temporal_embedding_dim: 256
embedding_init: "zero"
fourier_scale: 0.0
iterations: 80000
position_lr_init: 0.00016
# ... and 20+ more parameters
```

#### **3. System Information (`wandb-metadata.json`)**
Hardware and environment details:
```json
{
  "host": "cipollino.cip.ifi.lmu.de",
  "gpu": "NVIDIA GeForce RTX 2060 SUPER",
  "cpu_count": 8,
  "memory": {"total": 62.64},
  "python": "3.7.16",
  "git": {"commit": "ba3e3b7a28fd..."},
  "args": ["-s", "data/cut_roasted_beef", "--gdim", "32", ...]
}
```

#### **4. Code Reproducibility**
- **Exact code snapshot**: `code/train.py` (exact version used)
- **Environment**: `conda-environment.yaml` (complete conda env)
- **Dependencies**: `requirements.txt` (Python packages)
- **Code changes**: `diff.patch` (changes from last git commit)

### 🔍 Useful Wandb Analysis Commands

#### **Find Recent Experiments:**
```bash
# List recent runs (newest first)
ls -t wandb/run-*/

# Count total experiments
ls wandb/run-*/ | wc -l

# Find experiments from today
ls wandb/run-$(date +%Y%m%d)*/ 2>/dev/null || echo "No runs today"
```

#### **Extract Final Metrics:**
```bash
# Get final PSNR from recent runs
for run in $(ls -t wandb/run-*/files/wandb-summary.json | head -5); do
    echo "=== $(basename $(dirname $(dirname $run))) ==="
    cat $run | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'PSNR: {data.get(\"test/PSNR\", \"N/A\"):.2f}')
print(f'SSIM: {data.get(\"test/SSIM\", \"N/A\"):.3f}')
print(f'LPIPS: {data.get(\"test/LPIPS\", \"N/A\"):.3f}')
print(f'Gaussians: {data.get(\"Number of Gaussians\", \"N/A\")}')
print(f'Runtime: {data.get(\"_runtime\", \"N/A\"):.1f}s')
print()
"
done
```

#### **Compare Experiment Parameters:**
```bash
# Compare gaussian dimensions across recent runs
echo "Gaussian Embedding Dimensions in Recent Runs:"
for run in $(ls -t wandb/run-*/files/config.yaml | head -10); do
    gdim=$(grep "gaussian_embedding_dim:" $run | awk '{print $2}')
    experiment=$(grep "experiment_name:" $run | awk '{print $2}' | tr -d '"')
    echo "$experiment -> gdim: $gdim"
done
```

#### **Find Best Performing Experiments:**
```bash
# Find runs with highest PSNR
echo "Top 5 PSNR Results:"
for run in $(ls wandb/run-*/files/wandb-summary.json); do
    psnr=$(cat $run | python3 -c "import json, sys; print(json.load(sys.stdin).get('test/PSNR', 0))" 2>/dev/null || echo "0")
    echo "$psnr $run"
done | sort -nr | head -5 | while read psnr path; do
    run_id=$(basename $(dirname $(dirname $path)))
    echo "PSNR: $psnr - Run: $run_id"
done
```

### 📈 Integration with Wandb Dashboard

The local `wandb/` data syncs with your **online Wandb dashboard**:

- **Local**: Detailed files, logs, and exact reproducibility info
- **Online**: Interactive plots, metrics visualization, run comparisons
- **Dashboard URL**: https://wandb.ai/harjes-ludwig-maximilianuniversity-of-munich/E-D3DGS

**Benefits of Local Wandb Data:**
- ✅ **Complete reproducibility** - exact code, environment, and parameters
- ✅ **Offline analysis** - query results without internet connection  
- ✅ **Detailed debugging** - access to internal logs and metadata
- ✅ **Code snapshots** - see exact version used for any experiment
- ✅ **Environment tracking** - conda/pip dependencies for each run
- ✅ **System profiling** - hardware specs and resource usage

### 🎯 Common Wandb Workflows

#### **Reproduce a Specific Run:**
```bash
# Find the run you want to reproduce
RUN_ID="run-20250702_173239-ffel0omj"

# Check the exact parameters used
cat wandb/$RUN_ID/files/config.yaml

# Check the exact command line
cat wandb/$RUN_ID/files/wandb-metadata.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
print('Exact command:', ' '.join(data['args']))
"

# Check if code was modified
if [ -f wandb/$RUN_ID/files/diff.patch ]; then
    echo "Code was modified from git commit:"
    cat wandb/$RUN_ID/files/diff.patch
fi
```

#### **Debug Failed Experiments:**
```bash
# Check logs for errors
tail -50 wandb/$RUN_ID/logs/debug.log

# Check system resources during run
cat wandb/$RUN_ID/files/wandb-metadata.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'GPU: {data[\"gpu\"]}')
print(f'Memory: {data[\"memory\"][\"total\"]:.1f} GB')
print(f'Disk space: {data[\"disk\"][\"/\"][\"total\"]:.1f} GB')
"
```

#### **Track Parameter Sweeps:**
```bash
# Find all experiments with different gaussian dimensions
echo "Gaussian Dimension Sweep Results:"
for gdim in 4 8 16 32 64; do
    echo "=== GDIM $gdim ==="
    for run in wandb/run-*/files/config.yaml; do
        if grep -q "gaussian_embedding_dim: $gdim" $run; then
            summary_file=$(dirname $run)/wandb-summary.json
            if [ -f $summary_file ]; then
                psnr=$(cat $summary_file | python3 -c "import json,sys; print(f'{json.load(sys.stdin).get(\"test/PSNR\", 0):.2f}')" 2>/dev/null)
                echo "  PSNR: $psnr ($(basename $(dirname $(dirname $run))))"
            fi
        fi
    done
done
```

**💡 The `wandb/` folder is your local experiment database** - use it for detailed analysis, debugging, and ensuring complete reproducibility of your E-D3DGS research! 🚀

## 🚨 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Job pending on Abakus | Check `squeue -p Abaki` - nodes may be full. Try NvidiaAll or wait |
| "Memory specification cannot be satisfied" | Fixed for Abakus - update script if needed |
| "QoS abaki" access denied | You're authorized - check partition spelling |
| Job killed after 4 hours | Increase time: `--time 8:00:00` or use reservation |
| Conda environment not found | Run `conda activate ed3dgs` |
| Wandb not logging | Check `source bin/setup_wandb_team.sh` |
| Wandb offline/no internet | Local data still tracked in `wandb/` folder |
| Missing wandb data | Check `wandb/run-*/files/` for experiment files |
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

# Scientific comparison: Different initialization methods
./bin/run_experiment.sh --scene cut_roasted_beef --embedding_init xavier --gdim 32
./bin/run_experiment.sh --scene cut_roasted_beef --embedding_init fourier --fourier_scale 2.0 --gdim 32
./bin/run_experiment.sh --scene cut_roasted_beef --embedding_init structured_fourier --fourier_scale 1.0 --num_freq_bands 5 --gdim 32

# Frequency band ablation study
./bin/run_experiment.sh --scene cook_spinach --embedding_init structured_fourier --num_freq_bands 3 --gdim 32
./bin/run_experiment.sh --scene cook_spinach --embedding_init structured_fourier --num_freq_bands 6 --gdim 32
./bin/run_experiment.sh --scene cook_spinach --embedding_init structured_fourier --num_freq_bands 10 --gdim 64

# Temporal stability experiment
./bin/run_experiment.sh --scene vrig-chicken --temporal_embedding_init sinusoidal --tdim 512

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

## 🎯 **Key Takeaways & Decision Guide**

### 🧠 **The Fundamental Choice: Fourier vs Higher Dimensions**

**When to use Fourier methods:**
- ✅ Scenes with fine geometric details (textures, edges, complex surfaces)
- ✅ You want better high-frequency representation from the start
- ✅ You prefer interpretable frequency control (`structured_fourier`)
- ✅ You're comparing against NeRF-style methods

**When to use higher normal dimensions:**
- ✅ Scene complexity is unknown
- ✅ You want maximum model flexibility
- ✅ You prefer simple, well-tested approaches
- ✅ Memory/speed is not a major constraint

### ⚖️ **Equivalent Capacity Quick Reference**

For roughly the same representational power:
```bash
# ~16 effective dimensions
--embedding_init xavier --gdim 16                               # Flexible
--embedding_init fourier --fourier_scale 1.0 --gdim 16         # Better high-freq
--embedding_init structured_fourier --num_freq_bands 2 --gdim 16  # Interpretable

# ~32 effective dimensions  
--embedding_init xavier --gdim 32                               # Flexible
--embedding_init fourier --fourier_scale 2.0 --gdim 32         # Better high-freq
--embedding_init structured_fourier --num_freq_bands 5 --gdim 32  # Interpretable

# ~64 effective dimensions
--embedding_init xavier --gdim 64                               # Flexible  
--embedding_init fourier --fourier_scale 4.0 --gdim 64         # Better high-freq
--embedding_init structured_fourier --num_freq_bands 10 --gdim 64 # Interpretable
```

### 🚀 **Getting Started Recommendations**

**First-time users:**
```bash
./bin/run_experiment.sh --scene cut_roasted_beef --embedding_init xavier --gdim 32
```

**Want better detail:**
```bash
./bin/run_experiment.sh --scene cut_roasted_beef --embedding_init fourier --fourier_scale 2.0 --gdim 32
```

**Want systematic control:**
```bash
./bin/run_experiment.sh --scene cut_roasted_beef --embedding_init structured_fourier --num_freq_bands 5 --gdim 32
```

**Research/ablations:**
- Start with `xavier` baseline
- Compare against `fourier` with different scales
- Use `structured_fourier` for frequency-specific studies

**🎉 Ready to run E-D3DGS experiments with comprehensive embedding initialization and SLURM integration!** Start with the Quick Start section and monitor your experiments on Wandb! 🚀 