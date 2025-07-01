#!/usr/bin/env python3
"""
Wandb Cache Management for E-D3DGS
Prevents cache-related crashes during training
"""

import os
import shutil
import subprocess
import psutil
from pathlib import Path


def get_cache_size(cache_dir):
    """Get the size of wandb cache directory."""
    if not os.path.exists(cache_dir):
        return 0
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(cache_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
            except (OSError, FileNotFoundError):
                pass
    return total_size


def format_bytes(bytes_size):
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def clean_wandb_cache():
    """Clean wandb cache directory."""
    cache_dirs = [
        os.path.expanduser("~/.cache/wandb"),
        os.path.expanduser("~/.local/share/wandb/artifacts"),
        "./wandb"  # Local wandb directory
    ]
    
    total_cleaned = 0
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            print(f"🧹 Cleaning {cache_dir}")
            cache_size = get_cache_size(cache_dir)
            
            try:
                if cache_dir.endswith("wandb") and not cache_dir.startswith("./"):
                    # Clean cache but preserve important files
                    artifacts_dir = os.path.join(cache_dir, "artifacts")
                    if os.path.exists(artifacts_dir):
                        shutil.rmtree(artifacts_dir)
                        print(f"   Cleaned artifacts: {format_bytes(cache_size)}")
                        total_cleaned += cache_size
                elif cache_dir == "./wandb":
                    # Clean local wandb runs older than 7 days
                    current_time = os.path.getctime
                    week_ago = current_time(cache_dir) - (7 * 24 * 60 * 60)
                    
                    for item in os.listdir(cache_dir):
                        item_path = os.path.join(cache_dir, item)
                        if os.path.isdir(item_path) and item.startswith("run-"):
                            if os.path.getctime(item_path) < week_ago:
                                dir_size = get_cache_size(item_path)
                                shutil.rmtree(item_path)
                                total_cleaned += dir_size
                                print(f"   Removed old run: {item}")
                else:
                    shutil.rmtree(cache_dir)
                    print(f"   Cleaned: {format_bytes(cache_size)}")
                    total_cleaned += cache_size
                    
            except PermissionError:
                print(f"   ⚠️  Permission denied cleaning {cache_dir}")
            except Exception as e:
                print(f"   ⚠️  Error cleaning {cache_dir}: {e}")
    
    print(f"✅ Total space cleaned: {format_bytes(total_cleaned)}")
    return total_cleaned


def check_disk_space():
    """Check available disk space."""
    home_dir = os.path.expanduser("~")
    usage = shutil.disk_usage(home_dir)
    
    total = usage.total
    used = usage.used
    free = usage.free
    
    print(f"💾 Disk Space Status:")
    print(f"   Total: {format_bytes(total)}")
    print(f"   Used:  {format_bytes(used)} ({used/total*100:.1f}%)")
    print(f"   Free:  {format_bytes(free)} ({free/total*100:.1f}%)")
    
    # Warning if less than 10GB free
    if free < 10 * 1024**3:
        print(f"⚠️  WARNING: Low disk space! Only {format_bytes(free)} remaining")
        return False
    
    return True


def set_wandb_cache_dir():
    """Set wandb cache to a location with more space."""
    # Use the project directory for wandb cache
    project_cache = "./wandb_cache"
    os.makedirs(project_cache, exist_ok=True)
    
    # Set environment variables
    os.environ['WANDB_CACHE_DIR'] = project_cache
    os.environ['WANDB_DATA_DIR'] = project_cache
    
    print(f"📁 Set wandb cache directory to: {project_cache}")
    return project_cache


def monitor_wandb_process():
    """Monitor wandb processes and their memory usage."""
    wandb_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
        try:
            if 'wandb' in proc.info['name'].lower():
                wandb_processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if wandb_processes:
        print(f"🔄 Active wandb processes:")
        for proc in wandb_processes:
            memory_mb = proc['memory_info'].rss / 1024**2
            print(f"   PID {proc['pid']}: {memory_mb:.1f} MB")
    
    return wandb_processes


def main():
    """Main function for wandb cache management."""
    print("🚀 E-D3DGS Wandb Cache Manager")
    print("=" * 40)
    
    # Check disk space
    check_disk_space()
    
    # Check current cache size
    cache_dirs = [
        os.path.expanduser("~/.cache/wandb"),
        os.path.expanduser("~/.local/share/wandb"),
        "./wandb"
    ]
    
    total_cache_size = 0
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            size = get_cache_size(cache_dir)
            total_cache_size += size
            print(f"📁 {cache_dir}: {format_bytes(size)}")
    
    print(f"📊 Total wandb cache: {format_bytes(total_cache_size)}")
    
    # Clean if cache is large (>1GB)
    if total_cache_size > 1024**3:
        print(f"🧹 Cache is large ({format_bytes(total_cache_size)}), cleaning...")
        clean_wandb_cache()
    
    # Set up better cache location
    set_wandb_cache_dir()
    
    # Monitor processes
    monitor_wandb_process()
    
    print("✅ Wandb cache management complete!")


if __name__ == "__main__":
    main() 