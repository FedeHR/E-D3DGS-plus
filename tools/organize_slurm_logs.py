#!/usr/bin/env python3
"""
SLURM Log Organization Tool
Reorganizes and cleans up SLURM logs for better experiment tracking
"""

import os
import re
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import argparse
from collections import defaultdict


class SlurmLogOrganizer:
    def __init__(self, base_dir="experiments/slurm_logs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def parse_log_filename(self, filename):
        """Parse SLURM log filename to extract metadata"""
        # Pattern: dataset_scene_timestamp_jobid.ext
        # Example: dynerf_cut_roasted_beef_20250617_002820_12712.err
        pattern = r'([^_]+)_([^_]+_[^_]+)_(\d{8}_\d{6})_(\d+)\.(err|out)'
        match = re.match(pattern, filename)
        
        if match:
            dataset, scene, timestamp, job_id, ext = match.groups()
            return {
                'dataset': dataset,
                'scene': scene,
                'timestamp': timestamp,
                'job_id': job_id,
                'extension': ext,
                'datetime': datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
            }
        
        # Alternative pattern for embedding experiments
        # Example: embedding_fourier_scale4.0_cut_roasted_beef_12685.err
        pattern2 = r'embedding_([^_]+)_([^_]+)_(\d+)\.(err|out)'
        match2 = re.match(pattern2, filename)
        
        if match2:
            exp_type, scene, job_id, ext = match2.groups()
            return {
                'dataset': 'embedding',
                'scene': scene,
                'experiment_type': exp_type,
                'timestamp': None,
                'job_id': job_id,
                'extension': ext,
                'datetime': None
            }
            
        return None
    
    def organize_by_experiment(self, dry_run=False):
        """Organize logs by experiment type and date"""
        print("🗂️  Organizing SLURM logs by experiment...")
        
        # Create organized structure
        organized_dir = self.base_dir / "organized"
        if not dry_run:
            organized_dir.mkdir(exist_ok=True)
        
        # Scan all log files
        log_files = []
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.err') or file.endswith('.out'):
                    file_path = Path(root) / file
                    metadata = self.parse_log_filename(file)
                    if metadata:
                        log_files.append((file_path, metadata))
        
        # Group by experiment patterns
        experiments = defaultdict(list)
        
        for file_path, metadata in log_files:
            if metadata['dataset'] == 'embedding':
                exp_name = f"embedding_{metadata['experiment_type']}_{metadata['scene']}"
            else:
                exp_name = f"{metadata['dataset']}_{metadata['scene']}"
            
            experiments[exp_name].append((file_path, metadata))
        
        # Create organized structure
        summary = {}
        for exp_name, files in experiments.items():
            exp_dir = organized_dir / exp_name
            if not dry_run:
                exp_dir.mkdir(exist_ok=True)
            
            print(f"\n📁 {exp_name} ({len(files)} files)")
            
            # Sort by job ID or timestamp
            files.sort(key=lambda x: x[1]['job_id'])
            
            exp_summary = {
                'experiment': exp_name,
                'total_files': len(files),
                'jobs': []
            }
            
            for file_path, metadata in files:
                job_id = metadata['job_id']
                
                # Check if this is a pair (err + out)
                base_name = f"job_{job_id}"
                if metadata['timestamp']:
                    base_name = f"job_{job_id}_{metadata['timestamp']}"
                
                # New organized filename
                new_name = f"{base_name}.{metadata['extension']}"
                new_path = exp_dir / new_name
                
                if not dry_run:
                    if not new_path.exists():
                        shutil.copy2(file_path, new_path)
                        print(f"  📄 {file_path.name} → {new_name}")
                    else:
                        print(f"  ⚠️  {new_name} already exists, skipping")
                else:
                    print(f"  📄 Would move: {file_path.name} → {new_name}")
                
                # Check file size and status
                file_size = file_path.stat().st_size if file_path.exists() else 0
                
                job_info = {
                    'job_id': job_id,
                    'timestamp': metadata.get('timestamp'),
                    'file_type': metadata['extension'],
                    'size_bytes': file_size,
                    'original_path': str(file_path),
                    'new_path': str(new_path)
                }
                
                exp_summary['jobs'].append(job_info)
            
            summary[exp_name] = exp_summary
        
        # Save summary
        if not dry_run:
            summary_file = organized_dir / "organization_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\n📊 Summary saved to: {summary_file}")
        
        return summary
    
    def identify_crashed_jobs(self):
        """Identify jobs that crashed with memory issues"""
        print("🔍 Identifying crashed jobs...")
        
        crashed_jobs = []
        
        # Look for error files with memory-related issues
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.err'):
                    file_path = Path(root) / file
                    metadata = self.parse_log_filename(file)
                    
                    if metadata and file_path.stat().st_size > 1000:  # Only check non-empty files
                        try:
                            with open(file_path, 'r') as f:
                                content = f.read()
                            
                            # Check for memory issues
                            memory_issues = []
                            if 'CUDA out of memory' in content:
                                memory_issues.append('CUDA OOM')
                            if 'RuntimeError: out of memory' in content:
                                memory_issues.append('Runtime OOM')
                            if 'killed' in content.lower():
                                memory_issues.append('Process killed')
                            if 'segmentation fault' in content.lower():
                                memory_issues.append('Segmentation fault')
                            
                            # Extract training progress if available
                            progress_lines = [line for line in content.split('\n') if 'Training progress:' in line]
                            last_progress = progress_lines[-1] if progress_lines else None
                            
                            if memory_issues or (progress_lines and len(progress_lines) > 10):
                                crashed_jobs.append({
                                    'file': file_path,
                                    'metadata': metadata,
                                    'memory_issues': memory_issues,
                                    'last_progress': last_progress,
                                    'progress_lines': len(progress_lines),
                                    'file_size': file_path.stat().st_size
                                })
                                
                        except Exception as e:
                            print(f"  ⚠️  Error reading {file_path}: {e}")
        
        return crashed_jobs
    
    def clean_old_logs(self, days=7, dry_run=False):
        """Clean up logs older than specified days"""
        print(f"🧹 Cleaning logs older than {days} days...")
        
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned_files = []
        
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                file_path = Path(root) / file
                
                # Skip organized directory
                if 'organized' in str(file_path):
                    continue
                
                try:
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_date:
                        if not dry_run:
                            file_path.unlink()
                        cleaned_files.append(str(file_path))
                        print(f"  🗑️  {'Would remove' if dry_run else 'Removed'}: {file_path.name}")
                except Exception as e:
                    print(f"  ⚠️  Error processing {file_path}: {e}")
        
        return cleaned_files
    
    def generate_report(self):
        """Generate a comprehensive report of experiment status"""
        print("📊 Generating experiment report...")
        
        crashed_jobs = self.identify_crashed_jobs()
        
        # Group by experiment
        crash_summary = defaultdict(list)
        for job in crashed_jobs:
            if job['metadata']['dataset'] == 'embedding':
                exp_name = f"embedding_{job['metadata']['experiment_type']}_{job['metadata']['scene']}"
            else:
                exp_name = f"{job['metadata']['dataset']}_{job['metadata']['scene']}"
            crash_summary[exp_name].append(job)
        
        print("\n🚨 CRASH REPORT")
        print("=" * 50)
        
        for exp_name, jobs in crash_summary.items():
            print(f"\n📁 {exp_name}")
            print(f"   💥 {len(jobs)} crashed jobs")
            
            for job in jobs:
                job_id = job['metadata']['job_id']
                timestamp = job['metadata'].get('timestamp', 'unknown')
                progress = job['progress_lines']
                issues = ', '.join(job['memory_issues']) if job['memory_issues'] else 'Training stopped'
                
                print(f"   📄 Job {job_id} ({timestamp}): {progress} steps, Issues: {issues}")
                if job['last_progress']:
                    # Extract step count and metrics
                    progress_match = re.search(r'(\d+)/80000.*Loss=([0-9.]+).*psnr=([0-9.]+)', job['last_progress'])
                    if progress_match:
                        steps, loss, psnr = progress_match.groups()
                        print(f"      🎯 Last: Step {steps}/80000, Loss={loss}, PSNR={psnr}")
        
        return crash_summary


def main():
    parser = argparse.ArgumentParser(description="Organize and analyze SLURM logs")
    parser.add_argument('--organize', action='store_true', help='Organize logs by experiment')
    parser.add_argument('--report', action='store_true', help='Generate crash report')
    parser.add_argument('--clean', type=int, metavar='DAYS', help='Clean logs older than N days')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without doing it')
    
    args = parser.parse_args()
    
    organizer = SlurmLogOrganizer()
    
    if args.organize:
        organizer.organize_by_experiment(dry_run=args.dry_run)
    
    if args.report:
        organizer.generate_report()
    
    if args.clean:
        organizer.clean_old_logs(days=args.clean, dry_run=args.dry_run)
    
    if not any([args.organize, args.report, args.clean]):
        print("🔧 SLURM Log Organizer")
        print("Usage: python organize_slurm_logs.py [--organize] [--report] [--clean DAYS] [--dry-run]")
        print("\nAvailable actions:")
        print("  --organize    Reorganize logs by experiment type")
        print("  --report      Generate crash report and analysis")
        print("  --clean DAYS  Remove logs older than DAYS")
        print("  --dry-run     Preview actions without executing")


if __name__ == "__main__":
    main() 