#!/usr/bin/env python3
"""
Training Monitor Script for E-D3DGS
Monitors training progress and automatically restarts stuck processes
"""

import os
import time
import psutil
import subprocess
import argparse
from pathlib import Path
import json

def check_training_progress(model_path, max_stall_time=3600):
    """
    Check if training is making progress by monitoring file timestamps
    Returns: (is_stuck, last_update_time, stall_duration)
    """
    if not os.path.exists(model_path):
        return False, None, 0
    
    # Check training_time.txt for recent updates
    training_time_file = os.path.join(model_path, "training_time.txt")
    wandb_summary = None
    
    # Find wandb summary file
    wandb_dir = "wandb"
    if os.path.exists(wandb_dir):
        for item in os.listdir(wandb_dir):
            if item.startswith("run-") and os.path.isdir(os.path.join(wandb_dir, item)):
                summary_file = os.path.join(wandb_dir, item, "files", "wandb-summary.json")
                if os.path.exists(summary_file):
                    wandb_summary = summary_file
                    break
    
    # Get the most recent timestamp
    last_update = 0
    if os.path.exists(training_time_file):
        last_update = max(last_update, os.path.getmtime(training_time_file))
    
    if wandb_summary and os.path.exists(wandb_summary):
        last_update = max(last_update, os.path.getmtime(wandb_summary))
    
    if last_update == 0:
        return False, None, 0
    
    current_time = time.time()
    stall_duration = current_time - last_update
    is_stuck = stall_duration > max_stall_time
    
    return is_stuck, last_update, stall_duration

def find_training_process():
    """Find running training processes"""
    training_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python' and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'train.py' in cmdline or 'train_fourier.py' in cmdline:
                    training_processes.append({
                        'pid': proc.info['pid'],
                        'cmdline': cmdline,
                        'process': proc
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return training_processes

def kill_training_process(pid):
    """Safely kill a training process"""
    try:
        proc = psutil.Process(pid)
        print(f"🔪 Killing stuck training process {pid}...")
        proc.terminate()
        
        # Wait for graceful termination
        try:
            proc.wait(timeout=30)
        except psutil.TimeoutExpired:
            print(f"🔪 Force killing process {pid}...")
            proc.kill()
        
        return True
    except psutil.NoSuchProcess:
        print(f"Process {pid} already terminated")
        return True
    except Exception as e:
        print(f"Failed to kill process {pid}: {e}")
        return False

def restart_training(original_cmdline, model_path):
    """Restart training from the last checkpoint"""
    print(f"🔄 Restarting training from checkpoint...")
    
    # Find the latest checkpoint
    checkpoint_files = []
    point_cloud_dir = os.path.join(model_path, "point_cloud")
    if os.path.exists(point_cloud_dir):
        for item in os.listdir(point_cloud_dir):
            if item.startswith("iteration_"):
                try:
                    iter_num = int(item.split("_")[1])
                    checkpoint_path = os.path.join(point_cloud_dir, item, "deformation.pth")
                    if os.path.exists(checkpoint_path):
                        checkpoint_files.append((iter_num, checkpoint_path))
                except:
                    continue
    
    if not checkpoint_files:
        print("❌ No checkpoints found, cannot restart")
        return False
    
    # Use the latest checkpoint
    checkpoint_files.sort(key=lambda x: x[0], reverse=True)
    latest_checkpoint = checkpoint_files[0][1]
    
    # Modify the command line to include checkpoint
    cmd_parts = original_cmdline.split()
    
    # Add checkpoint argument if not already present
    if '--start_checkpoint' not in cmd_parts:
        cmd_parts.extend(['--start_checkpoint', latest_checkpoint])
    
    # Add restart suffix to experiment name
    if '--expname' in cmd_parts:
        expname_idx = cmd_parts.index('--expname') + 1
        if expname_idx < len(cmd_parts):
            cmd_parts[expname_idx] += f"_restart_{int(time.time())}"
    
    print(f"🚀 Restarting with command: {' '.join(cmd_parts)}")
    
    # Start the new process
    try:
        subprocess.Popen(cmd_parts, cwd=os.getcwd())
        return True
    except Exception as e:
        print(f"❌ Failed to restart training: {e}")
        return False

def monitor_training(model_path, max_stall_time=3600, check_interval=300, auto_restart=False):
    """
    Monitor training progress and optionally restart if stuck
    
    Args:
        model_path: Path to the model output directory
        max_stall_time: Maximum time (seconds) without progress before considering stuck
        check_interval: How often to check (seconds)
        auto_restart: Whether to automatically restart stuck training
    """
    print(f"🔍 Starting training monitor for {model_path}")
    print(f"📊 Check interval: {check_interval}s, Max stall time: {max_stall_time}s")
    print(f"🔄 Auto-restart: {'Enabled' if auto_restart else 'Disabled'}")
    
    while True:
        try:
            # Check training progress
            is_stuck, last_update, stall_duration = check_training_progress(model_path, max_stall_time)
            
            if last_update:
                last_update_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_update))
                print(f"📈 Last update: {last_update_str} ({stall_duration:.0f}s ago)")
            
            if is_stuck:
                print(f"⚠️  Training appears stuck (no progress for {stall_duration:.0f}s)")
                
                # Find training processes
                training_procs = find_training_process()
                
                if training_procs:
                    print(f"🔍 Found {len(training_procs)} training process(es)")
                    
                    if auto_restart:
                        for proc_info in training_procs:
                            # Kill the stuck process
                            if kill_training_process(proc_info['pid']):
                                # Restart training
                                restart_training(proc_info['cmdline'], model_path)
                                break
                    else:
                        print("🛑 Auto-restart disabled. Manual intervention required.")
                        print("💡 To restart manually:")
                        for proc_info in training_procs:
                            print(f"   kill {proc_info['pid']}")
                            print(f"   {proc_info['cmdline']} --start_checkpoint <latest_checkpoint>")
                else:
                    print("🔍 No training processes found - training may have crashed")
            else:
                if last_update:
                    print(f"✅ Training is progressing normally")
                else:
                    print(f"🔍 No training files found yet")
            
            print(f"😴 Sleeping for {check_interval}s...")
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\n🛑 Monitor stopped by user")
            break
        except Exception as e:
            print(f"❌ Monitor error: {e}")
            time.sleep(check_interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor E-D3DGS training progress")
    parser.add_argument("--model_path", required=True, help="Path to model output directory")
    parser.add_argument("--max_stall_time", type=int, default=3600, 
                       help="Maximum time without progress before considering stuck (seconds)")
    parser.add_argument("--check_interval", type=int, default=300,
                       help="How often to check progress (seconds)")
    parser.add_argument("--auto_restart", action="store_true",
                       help="Automatically restart stuck training")
    
    args = parser.parse_args()
    
    monitor_training(
        model_path=args.model_path,
        max_stall_time=args.max_stall_time,
        check_interval=args.check_interval,
        auto_restart=args.auto_restart
    ) 