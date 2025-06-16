#!/usr/bin/env python3
"""
E-D3DGS Training Progress Monitor
Real-time monitoring of training progress from SLURM error logs
"""

import os
import re
import sys
import time
import glob
import argparse
from datetime import datetime
from pathlib import Path

class TrainingMonitor:
    def __init__(self, log_dir="experiments/slurm_logs"):
        self.log_dir = Path(log_dir)
        self.progress_pattern = re.compile(
            r'Training progress: (\d+)/(\d+) with Loss=([0-9.]+) psnr=([0-9.]+) point=(\d+)'
        )
        
    def find_error_logs(self, job_id=None):
        """Find error log files, optionally filtered by job ID"""
        if job_id:
            pattern = f"**/*_{job_id}.err"
        else:
            pattern = "**/*.err"
            
        error_logs = list(self.log_dir.glob(pattern))
        return sorted(error_logs, key=lambda x: x.stat().st_mtime, reverse=True)
    
    def parse_training_progress(self, log_file):
        """Parse training progress from error log file"""
        if not log_file.exists():
            return None
            
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            progress_lines = []
            for line in lines:
                match = self.progress_pattern.search(line)
                if match:
                    iteration, total, loss, psnr, points = match.groups()
                    progress_lines.append({
                        'iteration': int(iteration),
                        'total': int(total),
                        'loss': float(loss),
                        'psnr': float(psnr),
                        'points': int(points),
                        'percent': (int(iteration) / int(total)) * 100
                    })
                    
            return progress_lines
            
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
            return None
    
    def get_experiment_info(self, log_file):
        """Extract experiment information from log filename"""
        filename = log_file.name
        parts = filename.replace('.err', '').split('_')
        
        if len(parts) >= 4:
            dataset = parts[0]
            scene = '_'.join(parts[1:-2])  # Handle multi-word scene names
            timestamp = parts[-2] + '_' + parts[-1]
            
            return {
                'dataset': dataset,
                'scene': scene,
                'timestamp': timestamp,
                'full_name': filename.replace('.err', '')
            }
        
        return {'dataset': 'unknown', 'scene': 'unknown', 'timestamp': '', 'full_name': filename}
    
    def format_progress_line(self, progress, exp_info):
        """Format a single progress line for display"""
        bar_length = 20
        filled_length = int(bar_length * progress['percent'] / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        return (f"{exp_info['dataset']}/{exp_info['scene']:<20} "
                f"[{bar}] {progress['percent']:5.1f}% "
                f"({progress['iteration']:>5}/{progress['total']}) "
                f"Loss: {progress['loss']:.4f} "
                f"PSNR: {progress['psnr']:.2f} "
                f"Points: {progress['points']:>6}")
    
    def show_current_status(self, job_id=None):
        """Show current training status for all or specific job"""
        error_logs = self.find_error_logs(job_id)
        
        if not error_logs:
            print("No error log files found")
            return
            
        print("🚂 E-D3DGS Training Progress Monitor")
        print("=" * 80)
        print(f"{'Experiment':<30} {'Progress':<25} {'Metrics':<25}")
        print("-" * 80)
        
        for log_file in error_logs:
            exp_info = self.get_experiment_info(log_file)
            progress_data = self.parse_training_progress(log_file)
            
            if progress_data and len(progress_data) > 0:
                latest = progress_data[-1]
                print(self.format_progress_line(latest, exp_info))
            else:
                print(f"{exp_info['dataset']}/{exp_info['scene']:<20} "
                      f"{'[' + '░' * 20 + ']':<25} "
                      f"Initializing...")
    
    def watch_progress(self, job_id=None, interval=10):
        """Watch training progress in real-time"""
        print(f"👀 Watching training progress (updating every {interval}s)")
        print("Press Ctrl+C to stop")
        print()
        
        try:
            while True:
                # Clear screen
                os.system('clear' if os.name == 'posix' else 'cls')
                
                print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print()
                
                self.show_current_status(job_id)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
    
    def show_detailed_progress(self, job_id):
        """Show detailed progress history for a specific job"""
        error_logs = self.find_error_logs(job_id)
        
        if not error_logs:
            print(f"No error log found for job {job_id}")
            return
            
        log_file = error_logs[0]
        exp_info = self.get_experiment_info(log_file)
        progress_data = self.parse_training_progress(log_file)
        
        if not progress_data:
            print(f"No training progress found in {log_file}")
            return
            
        print(f"📊 Detailed Progress for Job {job_id}")
        print(f"Experiment: {exp_info['dataset']}/{exp_info['scene']}")
        print(f"Log file: {log_file}")
        print("=" * 80)
        
        print(f"{'Iteration':<10} {'Progress':<10} {'Loss':<10} {'PSNR':<8} {'Points':<8}")
        print("-" * 50)
        
        # Show last 20 progress entries
        recent_progress = progress_data[-20:] if len(progress_data) > 20 else progress_data
        
        for p in recent_progress:
            print(f"{p['iteration']:<10} {p['percent']:>6.1f}%    "
                  f"{p['loss']:<10.4f} {p['psnr']:<8.2f} {p['points']:<8}")
        
        if len(progress_data) > 20:
            print(f"\n... showing last 20 of {len(progress_data)} total progress entries")
        
        # Show summary
        if len(progress_data) > 1:
            first = progress_data[0]
            latest = progress_data[-1]
            
            print(f"\n📈 Training Summary:")
            print(f"Progress: {latest['percent']:.1f}% ({latest['iteration']}/{latest['total']})")
            print(f"Loss improvement: {first['loss']:.4f} → {latest['loss']:.4f} "
                  f"({((first['loss'] - latest['loss']) / first['loss'] * 100):+.1f}%)")
            print(f"PSNR improvement: {first['psnr']:.2f} → {latest['psnr']:.2f} "
                  f"({latest['psnr'] - first['psnr']:+.2f})")
            print(f"Points: {first['points']} → {latest['points']} "
                  f"({latest['points'] - first['points']:+d})")

def main():
    parser = argparse.ArgumentParser(description='E-D3DGS Training Progress Monitor')
    parser.add_argument('--job-id', '-j', type=str, help='Monitor specific job ID')
    parser.add_argument('--watch', '-w', action='store_true', help='Watch progress in real-time')
    parser.add_argument('--detailed', '-d', action='store_true', help='Show detailed progress history')
    parser.add_argument('--interval', '-i', type=int, default=10, help='Update interval for watch mode (seconds)')
    parser.add_argument('--log-dir', type=str, default='experiments/slurm_logs', help='Directory containing log files')
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.log_dir)
    
    if args.watch:
        monitor.watch_progress(args.job_id, args.interval)
    elif args.detailed:
        if not args.job_id:
            print("Error: --detailed requires --job-id")
            sys.exit(1)
        monitor.show_detailed_progress(args.job_id)
    else:
        monitor.show_current_status(args.job_id)

if __name__ == '__main__':
    main() 