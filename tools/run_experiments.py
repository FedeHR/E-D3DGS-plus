#!/usr/bin/env python3
"""
E-D3DGS Experiment Runner
Generates and submits SLURM jobs for multiple configurations
"""

import os
import json
import argparse
import itertools
from pathlib import Path
import subprocess
import time

class ExperimentRunner:
    def __init__(self, base_config_dir="experiments", slurm_dir="slurm_jobs"):
        self.base_config_dir = Path(base_config_dir)
        self.slurm_dir = Path(slurm_dir)
        self.base_config_dir.mkdir(exist_ok=True)
        self.slurm_dir.mkdir(exist_ok=True)
        
    def create_experiment_config(self, dataset, scene, config):
        """Create experiment configuration file"""
        config_name = self.generate_config_name(dataset, scene, config)
        config_file = self.base_config_dir / f"{config_name}.json"
        
        # Full configuration with all parameters
        full_config = {
            "dataset": dataset,
            "scene": scene,
            "experiment_name": config_name,
            "model_params": {
                "use_fourier_features": config.get("use_fourier_features", False),
                "fourier_scale": config.get("fourier_scale", 1.0),
            },
            "hidden_params": {
                "gaussian_embedding_dim": config.get("gaussian_embedding_dim", 32),
                "temporal_embedding_dim": config.get("temporal_embedding_dim", 256),
                "embedding_init": config.get("embedding_init", "random"),  # random, zero, xavier, fourier
            },
            "optimization_params": {
                # Add any optimization parameter overrides here
            },
            "slurm_params": {
                "partition": config.get("partition", "gpu"),
                "time": config.get("time", "48:00:00"),
                "mem": config.get("mem", "32G"),
                "gpus": config.get("gpus", 1),
                "cpus": config.get("cpus", 8),
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(full_config, f, indent=2)
        
        return config_file, config_name
    
    def generate_config_name(self, dataset, scene, config):
        """Generate experiment name following the convention: dataset/scene-gdimX-tdimY-fourierZ"""
        parts = [f"{dataset}/{scene}"]
        
        # Add gaussian embedding dimension
        gdim = config.get("gaussian_embedding_dim", 32)
        parts.append(f"gdim{gdim}")
        
        # Add temporal embedding dimension if different from default
        tdim = config.get("temporal_embedding_dim", 256)
        if tdim != 256:
            parts.append(f"tdim{tdim}")
        
        # Add embedding initialization type
        init_type = config.get("embedding_init", "random")
        if init_type != "random":
            parts.append(init_type)
        
        # Add Fourier features if enabled
        if config.get("use_fourier_features", False):
            fourier_scale = config.get("fourier_scale", 1.0)
            if fourier_scale == int(fourier_scale):
                parts.append(f"fourier{int(fourier_scale)}")
            else:
                parts.append(f"fourier{fourier_scale}")
        
        return "-".join(parts)
    
    def create_slurm_script(self, config_file, config_name):
        """Create SLURM script for the experiment"""
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        dataset = config["dataset"]
        scene = config["scene"]
        slurm_params = config["slurm_params"]
        
        # Determine the base training script and dataset path
        if dataset == "dynerf":
            dataset_path = "datasets"
            base_script = "train.py"
            config_path = f"arguments/dynerf/{scene}.py"
        elif dataset == "hypernerf":
            dataset_path = "datasets"
            base_script = "train.py"
            config_path = f"arguments/hypernerf/{scene}.py"
        elif dataset == "technicolor":
            dataset_path = "datasets"
            base_script = "train.py"
            config_path = f"arguments/technicolor/{scene}.py"
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # Note: Fourier features are now integrated into train.py
        # No need for separate script - use embedding_init parameter instead
        
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name={config_name.replace('/', '_')}
#SBATCH --partition={slurm_params["partition"]}
#SBATCH --time={slurm_params["time"]}
#SBATCH --mem={slurm_params["mem"]}
#SBATCH --gpus={slurm_params["gpus"]}
#SBATCH --cpus-per-task={slurm_params["cpus"]}
#SBATCH --output=slurm_logs/{config_name.replace('/', '_')}_%j.out
#SBATCH --error=slurm_logs/{config_name.replace('/', '_')}_%j.err

# Load environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ed3dgs

# Setup wandb environment
source setup_wandb_team.sh

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Create output directory
mkdir -p output/{dataset}/{config_name.replace('/', '_')}
mkdir -p slurm_logs

echo "Starting experiment: {config_name}"
echo "Dataset: {dataset}"
echo "Scene: {scene}"
echo "Config file: {config_file}"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Build training command
TRAIN_CMD="python {base_script} \\
    -s {dataset_path}/{scene} \\
    --model_path output/{dataset}/{config_name.replace('/', '_')} \\
    --expname '{config_name}' \\
    --configs {config_path} \\
    -r 2"

# Add Fourier features parameters if enabled
"""
        
        if config["model_params"]["use_fourier_features"]:
            slurm_script += f"""TRAIN_CMD="$TRAIN_CMD \\
    --use_fourier_features \\
    --fourier_scale {config['model_params']['fourier_scale']}"

"""
        
        # Add embedding dimension parameters
        gdim = config["hidden_params"]["gaussian_embedding_dim"]
        tdim = config["hidden_params"]["temporal_embedding_dim"]
        
        if gdim != 32 or tdim != 256:
            slurm_script += f"""# Note: Embedding dimensions need to be set in the config file or passed as arguments
# gaussian_embedding_dim: {gdim}
# temporal_embedding_dim: {tdim}

"""
        
        slurm_script += f"""
echo "Training command: $TRAIN_CMD"

# Run training with monitoring
$TRAIN_CMD

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully"
    
    # Run rendering
    echo "Starting rendering..."
    python render.py \\
        --model_path output/{dataset}/{config_name.replace('/', '_')} \\
        --skip_train \\
        --configs {config_path}
    
    # Run evaluation
    echo "Starting evaluation..."
    python metrics.py \\
        --model_path output/{dataset}/{config_name.replace('/', '_')}
    
    echo "Experiment {config_name} completed successfully"
else
    echo "Training failed with exit code $?"
    exit 1
fi
"""
        
        slurm_file = self.slurm_dir / f"{config_name.replace('/', '_')}.slurm"
        with open(slurm_file, 'w') as f:
            f.write(slurm_script)
        
        # Make script executable
        os.chmod(slurm_file, 0o755)
        
        return slurm_file
    
    def generate_experiments(self, datasets, scenes, parameter_grid):
        """Generate all experiment combinations"""
        experiments = []
        
        for dataset in datasets:
            for scene in scenes.get(dataset, []):
                for params in parameter_grid:
                    config_file, config_name = self.create_experiment_config(dataset, scene, params)
                    slurm_file = self.create_slurm_script(config_file, config_name)
                    
                    experiments.append({
                        "config_name": config_name,
                        "config_file": config_file,
                        "slurm_file": slurm_file,
                        "dataset": dataset,
                        "scene": scene,
                        "params": params
                    })
        
        return experiments
    
    def submit_experiments(self, experiments, dry_run=False, max_concurrent=None):
        """Submit experiments to SLURM"""
        if dry_run:
            print("DRY RUN - Would submit the following experiments:")
            for exp in experiments:
                print(f"  {exp['config_name']} -> {exp['slurm_file']}")
            return
        
        # Create slurm_logs directory
        os.makedirs("slurm_logs", exist_ok=True)
        
        submitted_jobs = []
        
        for i, exp in enumerate(experiments):
            if max_concurrent and len(submitted_jobs) >= max_concurrent:
                print(f"Reached maximum concurrent jobs ({max_concurrent}). Waiting...")
                # Wait for some jobs to complete
                time.sleep(60)
                # Check job status and remove completed jobs
                submitted_jobs = self.check_running_jobs(submitted_jobs)
            
            print(f"Submitting experiment {i+1}/{len(experiments)}: {exp['config_name']}")
            
            try:
                result = subprocess.run(
                    ["sbatch", str(exp['slurm_file'])],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Extract job ID from sbatch output
                job_id = result.stdout.strip().split()[-1]
                submitted_jobs.append({
                    "job_id": job_id,
                    "config_name": exp['config_name'],
                    "slurm_file": exp['slurm_file']
                })
                
                print(f"  Submitted job {job_id}")
                
            except subprocess.CalledProcessError as e:
                print(f"  Failed to submit: {e}")
                print(f"  Error output: {e.stderr}")
        
        print(f"\nSubmitted {len(submitted_jobs)} experiments")
        return submitted_jobs
    
    def check_running_jobs(self, job_list):
        """Check which jobs are still running"""
        if not job_list:
            return []
        
        try:
            result = subprocess.run(
                ["squeue", "-h", "-o", "%i"],
                capture_output=True,
                text=True,
                check=True
            )
            
            running_job_ids = set(result.stdout.strip().split())
            still_running = [job for job in job_list if job["job_id"] in running_job_ids]
            
            return still_running
            
        except subprocess.CalledProcessError:
            # If squeue fails, assume all jobs are still running
            return job_list

def create_parameter_grid():
    """Create parameter grid for experiments"""
    
    # Define parameter variations
    gaussian_dims = [8, 16, 32, 64]
    temporal_dims = [128, 256, 512]
    embedding_inits = ["random", "zero", "xavier"]
    fourier_configs = [
        {"use_fourier_features": False},
        {"use_fourier_features": True, "fourier_scale": 1.0},
        {"use_fourier_features": True, "fourier_scale": 2.0},
        {"use_fourier_features": True, "fourier_scale": 4.0},
    ]
    
    # Generate all combinations
    parameter_grid = []
    
    for gdim, tdim, init, fourier in itertools.product(
        gaussian_dims, temporal_dims, embedding_inits, fourier_configs
    ):
        config = {
            "gaussian_embedding_dim": gdim,
            "temporal_embedding_dim": tdim,
            "embedding_init": init,
            **fourier
        }
        parameter_grid.append(config)
    
    return parameter_grid

def main():
    parser = argparse.ArgumentParser(description="E-D3DGS Experiment Runner")
    parser.add_argument("--datasets", nargs="+", default=["dynerf"], 
                       choices=["dynerf", "hypernerf", "technicolor"],
                       help="Datasets to run experiments on")
    parser.add_argument("--scenes", type=str, help="JSON file with scene configurations")
    parser.add_argument("--config", type=str, help="JSON file with parameter grid")
    parser.add_argument("--dry_run", action="store_true", help="Generate scripts but don't submit")
    parser.add_argument("--max_concurrent", type=int, help="Maximum concurrent jobs")
    parser.add_argument("--submit", action="store_true", help="Submit jobs to SLURM")
    
    args = parser.parse_args()
    
    # Default scene configurations
    default_scenes = {
        "dynerf": ["cut_roasted_beef", "cook_spinach", "sear_steak"],
        "hypernerf": ["vrig-chicken", "vrig-broom", "vrig-3dprinter"],
        "technicolor": ["Painter", "Birthday", "Fabien"]
    }
    
    # Load scene configuration if provided
    if args.scenes:
        with open(args.scenes, 'r') as f:
            scenes = json.load(f)
    else:
        scenes = default_scenes
    
    # Load parameter grid if provided
    if args.config:
        with open(args.config, 'r') as f:
            parameter_grid = json.load(f)
    else:
        # Use a smaller default grid for testing
        parameter_grid = [
            {"gaussian_embedding_dim": 8, "temporal_embedding_dim": 256, "embedding_init": "random", "use_fourier_features": False},
            {"gaussian_embedding_dim": 32, "temporal_embedding_dim": 256, "embedding_init": "random", "use_fourier_features": False},
            {"gaussian_embedding_dim": 32, "temporal_embedding_dim": 256, "embedding_init": "random", "use_fourier_features": True, "fourier_scale": 4.0},
        ]
    
    # Create experiment runner
    runner = ExperimentRunner()
    
    # Generate experiments
    print(f"Generating experiments for datasets: {args.datasets}")
    experiments = runner.generate_experiments(args.datasets, scenes, parameter_grid)
    
    print(f"Generated {len(experiments)} experiments")
    
    if args.submit:
        # Submit experiments
        runner.submit_experiments(experiments, dry_run=args.dry_run, max_concurrent=args.max_concurrent)
    else:
        print("Use --submit to submit jobs to SLURM")
        print("Generated files:")
        for exp in experiments[:5]:  # Show first 5
            print(f"  Config: {exp['config_file']}")
            print(f"  SLURM:  {exp['slurm_file']}")
        if len(experiments) > 5:
            print(f"  ... and {len(experiments) - 5} more")

if __name__ == "__main__":
    main() 