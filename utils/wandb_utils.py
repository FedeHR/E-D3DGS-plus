"""
Wandb utilities for team collaboration and experiment tracking
"""
import wandb
import os
import json
from pathlib import Path


def setup_wandb_for_team(project_name=None, entity=None, run_name=None, config=None, tags=None):
    """
    Setup wandb with team-friendly defaults
    """
    # Use environment variables if not provided
    project = project_name or os.getenv('WANDB_PROJECT', 'E-D3DGS')
    entity = entity or os.getenv('WANDB_ENTITY', None)
    
    # Create run name with user info
    if not run_name:
        import socket, time
        hostname = socket.gethostname()
        username = os.getenv('USER', 'unknown')
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{username}_{hostname}_{timestamp}"
    
    # Add git info to config
    if config is None:
        config = {}
    
    config.update({
        "git_commit": os.popen('git rev-parse HEAD 2>/dev/null').read().strip(),
        "git_branch": os.popen('git rev-parse --abbrev-ref HEAD 2>/dev/null').read().strip(),
        "hostname": socket.gethostname(),
        "username": os.getenv('USER', 'unknown'),
    })
    
    # Default tags
    if tags is None:
        tags = [f"user_{config['username']}", f"host_{config['hostname']}"]
    
    return wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config,
        tags=tags
    )


def log_experiment_info(args, dataset, opt, hyper):
    """
    Log comprehensive experiment information
    """
    experiment_info = {
        "model": {
            "sh_degree": dataset.sh_degree,
            "gaussian_embedding_dim": hyper.gaussian_embedding_dim,
            "temporal_embedding_dim": hyper.temporal_embedding_dim,
            "net_width": hyper.net_width,
            "defor_depth": hyper.defor_depth,
        },
        "optimization": {
            "iterations": opt.iterations,
            "batch_size": opt.batch_size,
            "position_lr_init": opt.position_lr_init,
            "deformation_lr_init": opt.deformation_lr_init,
            "densify_grad_threshold": opt.densify_grad_threshold_fine_init,
            "opacity_threshold": opt.opacity_threshold_fine_init,
            "lambda_dssim": opt.lambda_dssim,
        },
        "dataset": {
            "source_path": dataset.source_path,
            "loader": dataset.loader,
            "white_background": dataset.white_background,
            "resolution": dataset._resolution,
        },
        "fourier_features": {
            "use_fourier": getattr(dataset, 'use_fourier_features', False),
            "fourier_scale": getattr(dataset, 'fourier_scale', 1.0),
        }
    }
    
    wandb.config.update(experiment_info)
    return experiment_info


def compare_runs(run_ids, metric="test/PSNR"):
    """
    Compare multiple wandb runs
    Usage: compare_runs(["run_id_1", "run_id_2"], "test/PSNR")
    """
    api = wandb.Api()
    
    runs_data = []
    for run_id in run_ids:
        run = api.run(f"{wandb.run.project}/{run_id}")
        runs_data.append({
            "id": run_id,
            "name": run.name,
            "config": run.config,
            "summary": run.summary,
            "final_metric": run.summary.get(metric, "N/A")
        })
    
    return runs_data


def create_experiment_summary(output_dir="./experiment_logs"):
    """
    Create a local summary of the current experiment
    """
    if not wandb.run:
        print("No active wandb run found")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    summary = {
        "run_id": wandb.run.id,
        "run_name": wandb.run.name,
        "project": wandb.run.project,
        "config": dict(wandb.config),
        "summary": dict(wandb.run.summary),
        "tags": wandb.run.tags,
        "url": wandb.run.url
    }
    
    summary_file = Path(output_dir) / f"experiment_{wandb.run.id}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Experiment summary saved to: {summary_file}")
    return summary_file


def log_fourier_comparison(baseline_metrics, fourier_metrics, step=None):
    """
    Log comparison between baseline and Fourier feature results
    """
    comparison = {}
    for metric in baseline_metrics:
        if metric in fourier_metrics:
            baseline_val = baseline_metrics[metric]
            fourier_val = fourier_metrics[metric]
            improvement = fourier_val - baseline_val
            comparison[f"comparison/{metric}_improvement"] = improvement
            comparison[f"comparison/{metric}_baseline"] = baseline_val
            comparison[f"comparison/{metric}_fourier"] = fourier_val
    
    wandb.log(comparison, step=step)
    return comparison 