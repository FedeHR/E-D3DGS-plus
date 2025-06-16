#  E-D3DGS : Embedding-Based Deformable 3D Gaussian Splatting (ECCV 2024)

[![arXiv](https://img.shields.io/badge/arXiv-2404.03613-006600)](https://arxiv.org/abs/2404.03613) 
[![project_page](https://img.shields.io/badge/project_page-68BC71)](https://jeongminb.github.io/e-d3dgs/)

[Jeongmin Bae](https://jeongminb.github.io/)<sup>1*</sup>, [Seoha Kim](https://seoha-kim.github.io/)<sup>1*</sup>, [Youngsik Yun](https://bbangsik13.github.io/)<sup>1</sup>, </br>
Hahyun Lee<sup>2 </sup>, Gun Bang<sup>2</sup>, [Youngjung Uh](https://github.com/yj-uh)<sup>1†</sup>

<sup>1</sup>Yonsei University &emsp; <sup>2</sup>Electronics and Telecommunications Research Institute (ETRI)
<br><sup>\*</sup> Equal Contributions &emsp; <sup>†</sup> Corresponding Author

---

Official repository for <a href="https://arxiv.org/abs/2404.03613">"Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting"</a>. Our approach employs per-Gaussian latent embeddings to predict deformation for each Gaussian and achieves a clearer representation of dynamic motion.

We uploaded the checkpoints, configs, and rendered videos for paper results [here](https://drive.google.com/drive/folders/1PAaIp5cNYNpLjQ5JX0SVLh5Yn_K9UmJd?usp=sharing).

![Alt Text](https://github.com/JeongminB/E-D3DGS/blob/main/teaser.gif)

## Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/JeongminB/E-D3DGS.git
cd E-D3DGS
git submodule update --init --recursive

conda create -n ed3dgs python=3.7 
conda activate ed3dgs

pip install -r requirements.txt
pip install -e submodules/diff-gaussian-rasterization/
pip install -e submodules/simple-knn/ 
```
We use `pytorch=1.13.1+cu116` in our environment.

### 2. Run an Experiment

**SLURM Cluster (Recommended):**
```bash
# Simple experiment - auto-detects dataset type
./bin/run_experiment.sh --scene cut_roasted_beef

# HyperNeRF scene
./bin/run_experiment.sh --scene vrig-chicken

# High-priority partition
./bin/run_experiment.sh --scene cut_roasted_beef --abakus

# Custom parameters
./bin/run_experiment.sh --scene cut_roasted_beef --gdim 64 --fourier_scale 4.0
```

**Local Execution (SSH Terminal):**
```bash
# Run directly in your terminal with real-time progress
./bin/run_experiment.sh --scene cut_roasted_beef --no_slurm

# This will show training progress live in your terminal
# Press Ctrl+C to stop the experiment
```

**Monitor Experiments:**
```bash
./tools/monitor_experiments.sh --status    # Show all running jobs
./tools/monitor_experiments.sh --logs      # Show latest logs  
./tools/monitor_experiments.sh --watch     # Watch logs in real-time
```

## Supported Datasets

The experiment runner automatically detects dataset types based on scene names:

**DyNeRF Scenes (Neural 3D Video):**
- `coffee_martini`, `cook_spinach`, `cut_roasted_beef`
- `flame_salmon_1`, `flame_steak`, `sear_steak`

**HyperNeRF Scenes:**
- `aleks-teapot`, `chickchicken`, `cut-lemon`, `hand`, `slice-banana`, `torchocolate`
- `americano`, `cross-hands`, `espresso`, `keyboard`, `oven-mitts`, `split-cookie`, `tamping`
- `3dprinter`, `broom`, `vrig-chicken`, `peel-banana`

## Data Preparation

**Downloading Datasets:**  
Please download datasets from their official websites: [HyperNerf](https://github.com/google/hypernerf/releases/tag/v0.1), [Neural 3D Video](https://github.com/facebookresearch/Neural_3D_Video) and [Technicolor](https://www.interdigital.com/data_sets/light-field-dataset)

- Please remove 'cam13.mp4' and corresponding pose from *coffee_martini* scene in the Neural 3D Video dataset
- We split the entire *flame_salmon_1_split* scene into four 300-frame scenes

**Extracting point clouds from COLMAP:** 
```bash
# setup COLMAP 
bash script/colmap_setup.sh
conda activate colmapenv 

# automatically extract the frames and reorganize them
python script/pre_n3v.py --videopath <dataset>/<scene>
python script/pre_technicolor.py --videopath <dataset>/<scene>
python script/pre_hypernerf.py --videopath <dataset>/<scene>

# downsample dense point clouds
python script/downsample_point.py \
<location>/<scene>/colmap/dense/workspace/fused.ply <location>/<scene>/points3D_downsample.ply
```

After running COLMAP, datasets should be organized as follows:
```
├── data
│   ├── cook_spinach
│   │   ├── colmap
│   │   ├── images
│   │   │   ├── cam01
│   │   │   │   ├── 0000.png
│   │   │   │   ├── 0001.png
│   │   │   │   └── ...
│   │   │   ├── cam02
│   │   │   └── ...
│   ├── cut_roasted_beef
│   └── ...
```

## Manual Training (Advanced)

For manual control or custom configurations:

```bash
# Training
python train.py -s data/$SCENE --configs arguments/$DATASET/$SCENE.py \
    --model_path results/$DATASET/$SCENE --expname $DATASET/$SCENE -r 2

# Rendering
python render.py --model_path results/$DATASET/$SCENE \
    --configs arguments/$DATASET/$SCENE.py --skip_train

# Evaluation
python metrics.py --model_path results/$DATASET/$SCENE
```

## Experiment Parameters

### Model Parameters
- **`--gdim`** (default: 32): Gaussian embedding dimension. Controls the dimensionality of per-Gaussian latent embeddings used for deformation prediction. Higher values may capture more complex deformations but increase memory usage.
- **`--tdim`** (default: 256): Temporal embedding dimension. Defines the size of time-dependent embeddings that encode temporal information for dynamic scenes.
- **`--fourier_scale`** (default: 0): Fourier features scale. When > 0, enables Fourier feature encoding for better high-frequency detail capture. Scale controls the frequency range.
- **`--embedding_init`** (default: random): Embedding initialization method (random, zero, xavier). Affects convergence speed and final quality.

### Training Parameters
- **`--resolution`** (default: 2): Resolution scaling factor. Higher values use higher resolution images but require more memory.
- **`--gpu`** (default: 0): GPU device ID for local execution.

### Execution Options
- **`--slurm`**: Submit to SLURM cluster (default behavior)
- **`--no_slurm`**: Run locally in terminal with real-time progress
- **`--abakus`**: Use high-priority Abaki partition with abaki QoS
- **`--partition`**: Specify SLURM partition (NvidiaAll, Abaki, All)
- **`--dry_run`**: Show commands without executing
- **`--skip_render`**: Skip rendering step
- **`--skip_eval`**: Skip evaluation step

## Experiment Tracking

We use Weights & Biases (wandb) for experiment tracking. The following parameters are automatically logged:

### Model Configuration
- **`sh_degree`**: Spherical harmonics degree for color representation
- **`gaussian_embedding_dim`**: Per-Gaussian embedding dimension for deformation
- **`temporal_embedding_dim`**: Temporal embedding dimension for time encoding
- **`net_width`**: Neural network width for deformation prediction
- **`defor_depth`**: Deformation network depth

### Optimization Settings
- **`iterations`**: Total training iterations
- **`position_lr_init`**: Initial learning rate for Gaussian positions
- **`deformation_lr_init`**: Initial learning rate for deformation parameters
- **`densify_grad_threshold`**: Gradient threshold for Gaussian densification
- **`opacity_threshold`**: Opacity threshold for Gaussian pruning
- **`lambda_dssim`**: SSIM loss weight

### Dataset Information
- **`source_path`**: Path to dataset
- **`loader`**: Dataset loader type (dynerf, hypernerf, technicolor)
- **`white_background`**: Whether to use white background
- **`resolution`**: Image resolution scaling

### Fourier Features
- **`use_fourier`**: Whether Fourier features are enabled
- **`fourier_scale`**: Fourier feature frequency scale

### System Information
- **`git_commit`**: Git commit hash for reproducibility
- **`git_branch`**: Git branch name
- **`hostname`**: Compute node hostname
- **`username`**: User running the experiment

## File Organization

Experiments are automatically organized by dataset type:
```
experiments/
├── slurm_jobs/
│   ├── dynerf/
│   │   └── dynerf_cut_roasted_beef_20250616_014021.sh
│   └── hypernerf/
│       └── hypernerf_vrig-chicken_20250616_013841.sh
└── slurm_logs/
    ├── dynerf/
    │   ├── dynerf_cut_roasted_beef_20250616_014021_12366.out
    │   └── dynerf_cut_roasted_beef_20250616_014021_progress.txt
    └── hypernerf/
        └── hypernerf_vrig-chicken_20250616_013841_progress.txt
```

## SLURM Configuration

The system supports multiple SLURM partitions:
- **NvidiaAll**: General NVIDIA GPU partition (default, most reliable)
- **Abaki**: High-priority partition (use `--abakus` flag, requires abaki QoS)
- **All**: General compute partition

Memory specifications are automatically handled to avoid SLURM errors.

## Troubleshooting

**Common Issues:**
1. **SLURM memory errors**: Use NvidiaAll partition (default) which doesn't require memory specification
2. **Dataset not found**: Ensure dataset is in `data/` directory with correct structure
3. **Stuck loading test cameras**: Large datasets may take time during initial evaluation; this is normal
4. **Permission denied**: Ensure scripts are executable: `chmod +x bin/run_experiment.sh`

**Getting Help:**
- Check experiment status: `./tools/monitor_experiments.sh --status`
- View logs: `./tools/monitor_experiments.sh --logs`
- Watch real-time: `./tools/monitor_experiments.sh --watch`

## Acknowledgements

This code is based on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [4DGaussians](https://github.com/hustvl/4DGaussians) and [STG](https://github.com/oppo-us-research/SpacetimeGaussians). In particular, we used [4DGaussians](https://github.com/hustvl/4DGaussians) as a starting point for our study. We would like to thank the authors of these papers for their hard work. 😊

## BibTex
```
@inproceedings{bae2024ed3dgs,
    title={Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting}, 
    author={Bae, Jeongmin and Kim, Seoha and Yun, Youngsik and Lee, Hahyun and Bang, Gun and Uh, Youngjung}, 
    booktitle = {European Conference on Computer Vision (ECCV)},
    year={2024}
}
```
