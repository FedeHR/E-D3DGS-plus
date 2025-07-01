#!/usr/bin/env python3
"""
Experiment Recovery Tool for E-D3DGS
Diagnoses and fixes common causes of experiment failures
"""

import os
import sys
import glob
import argparse
import subprocess
from pathlib import Path
from PIL import Image
import shutil


def find_failed_experiments():
    """Find experiments that failed due to corrupted images."""
    failed_experiments = []
    
    # Check slurm logs for image corruption errors
    log_dirs = [
        "experiments/slurm_logs",
        "experiments/slurm_logs/archive"
    ]
    
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            for err_file in glob.glob(f"{log_dir}/**/*.err", recursive=True):
                try:
                    with open(err_file, 'r') as f:
                        content = f.read()
                        if "unrecognized data stream contents" in content or "OSError" in content:
                            # Extract experiment name from path
                            exp_name = Path(err_file).parent.name
                            failed_experiments.append({
                                'name': exp_name,
                                'error_file': err_file,
                                'type': 'image_corruption'
                            })
                except Exception as e:
                    print(f"Could not read {err_file}: {e}")
    
    return failed_experiments


def diagnose_image_corruption(scene_name):
    """Diagnose image corruption issues for a specific scene."""
    scene_path = f"data/{scene_name}"
    
    if not os.path.exists(scene_path):
        print(f"❌ Scene path not found: {scene_path}")
        return False
    
    print(f"🔍 Diagnosing image corruption in: {scene_path}")
    
    # Find all image directories
    image_dirs = []
    images_dir = os.path.join(scene_path, "images")
    if os.path.exists(images_dir):
        for item in os.listdir(images_dir):
            item_path = os.path.join(images_dir, item)
            if os.path.isdir(item_path):
                image_dirs.append(item_path)
    
    corrupted_images = []
    total_images = 0
    
    for img_dir in image_dirs:
        print(f"   Checking {img_dir}...")
        png_files = glob.glob(os.path.join(img_dir, "*.png"))
        
        for img_path in png_files:
            total_images += 1
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                corrupted_images.append({
                    'path': img_path,
                    'error': str(e)
                })
                print(f"   ❌ {img_path}: {e}")
    
    print(f"📊 Summary:")
    print(f"   Total images: {total_images}")
    print(f"   Corrupted: {len(corrupted_images)}")
    
    return corrupted_images


def fix_corrupted_images(corrupted_images):
    """Attempt to fix corrupted images by replacing with nearby frames."""
    fixed_count = 0
    
    for corrupted in corrupted_images:
        img_path = corrupted['path']
        print(f"🔧 Attempting to fix: {img_path}")
        
        # Find replacement image
        replacement = find_replacement_image(img_path)
        if replacement:
            try:
                # Backup the corrupted file
                backup_path = img_path + ".corrupted_backup"
                shutil.move(img_path, backup_path)
                
                # Copy replacement
                shutil.copy2(replacement, img_path)
                print(f"   ✅ Replaced with: {replacement}")
                fixed_count += 1
                
            except Exception as e:
                print(f"   ❌ Failed to replace: {e}")
                # Restore backup if copy failed
                if os.path.exists(backup_path):
                    shutil.move(backup_path, img_path)
        else:
            print(f"   ❌ No suitable replacement found")
    
    print(f"🔧 Fixed {fixed_count} out of {len(corrupted_images)} corrupted images")
    return fixed_count


def find_replacement_image(img_path):
    """Find a suitable replacement for a corrupted image."""
    path_obj = Path(img_path)
    parent_dir = path_obj.parent
    filename = path_obj.stem
    extension = path_obj.suffix
    
    # Try to extract frame number
    try:
        frame_num = int(''.join(filter(str.isdigit, filename)))
        
        # Look for nearby frames (±10 frames)
        for offset in range(1, 11):
            for direction in [-1, 1]:
                candidate_num = frame_num + (offset * direction)
                if candidate_num >= 0:
                    candidate_path = parent_dir / f"{candidate_num:04d}{extension}"
                    if candidate_path.exists() and candidate_path != Path(img_path):
                        try:
                            with Image.open(candidate_path) as test_img:
                                test_img.verify()
                            return str(candidate_path)
                        except:
                            continue
    except ValueError:
        pass
    
    # If frame-based approach fails, find any valid image in directory
    for candidate_path in parent_dir.glob(f"*{extension}"):
        if candidate_path != Path(img_path):
            try:
                with Image.open(candidate_path) as test_img:
                    test_img.verify()
                return str(candidate_path)
            except:
                continue
    
    return None


def restart_experiment(exp_name, scene_name):
    """Restart a failed experiment with the same parameters."""
    print(f"🚀 Restarting experiment: {exp_name}")
    
    # Extract parameters from experiment name
    parts = exp_name.split('-')
    if len(parts) >= 2:
        # Try to extract gdim and tdim from name
        gdim = None
        tdim = None
        
        for part in parts:
            if part.startswith('gdim'):
                gdim = part[4:]
            elif part.startswith('tdim'):
                tdim = part[4:]
        
        # Build restart command
        cmd = ["./bin/run_experiment.sh", "--scene", scene_name]
        
        if gdim:
            cmd.extend(["--gdim", gdim])
        if tdim:
            cmd.extend(["--tdim", tdim])
        
        print(f"   Command: {' '.join(cmd)}")
        
        # Ask for confirmation
        response = input("   Restart this experiment? [y/N]: ").strip().lower()
        if response == 'y':
            subprocess.run(cmd)
        else:
            print("   Restart cancelled")
    else:
        print(f"   Could not parse experiment parameters from: {exp_name}")


def main():
    """Main recovery function."""
    parser = argparse.ArgumentParser(
        description="Recover from failed E-D3DGS experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python recover_failed_experiment.py                                    # Find and fix all failures
  python recover_failed_experiment.py --scene cut_roasted_beef          # Fix specific scene
  python recover_failed_experiment.py --experiment exp_name             # Fix specific experiment
        """
    )
    
    parser.add_argument(
        '--scene',
        help='Scene name to check and fix'
    )
    
    parser.add_argument(
        '--experiment',
        help='Specific experiment name to restart'
    )
    
    parser.add_argument(
        '--no-fix',
        action='store_true',
        help='Only diagnose, do not attempt to fix'
    )
    
    args = parser.parse_args()
    
    print("🚑 E-D3DGS Experiment Recovery Tool")
    print("=" * 40)
    
    if args.scene:
        # Check specific scene
        corrupted = diagnose_image_corruption(args.scene)
        if corrupted and not args.no_fix:
            fix_corrupted_images(corrupted)
    
    elif args.experiment:
        # Handle specific experiment
        print(f"🔍 Checking experiment: {args.experiment}")
        # Extract scene name from experiment name
        scene_name = args.experiment.split('-')[0]  # Assume first part is scene
        corrupted = diagnose_image_corruption(scene_name)
        if corrupted and not args.no_fix:
            fixed = fix_corrupted_images(corrupted)
            if fixed > 0:
                restart_experiment(args.experiment, scene_name)
    
    else:
        # Find all failed experiments
        failed = find_failed_experiments()
        print(f"🔍 Found {len(failed)} failed experiments")
        
        for exp in failed:
            print(f"\n📁 {exp['name']} ({exp['type']})")
            print(f"   Error log: {exp['error_file']}")
            
            if exp['type'] == 'image_corruption':
                # Extract scene name
                scene_name = exp['name'].split('-')[0]
                corrupted = diagnose_image_corruption(scene_name)
                
                if corrupted and not args.no_fix:
                    fixed = fix_corrupted_images(corrupted)
                    if fixed > 0:
                        restart_experiment(exp['name'], scene_name)
    
    print("\n✅ Recovery process complete!")


if __name__ == "__main__":
    main() 