#!/usr/bin/env python3
"""
Image integrity checker for E-D3DGS datasets
Scans dataset directories for corrupted or problematic images
"""

import os
import sys
import glob
import argparse
from PIL import Image
from pathlib import Path

def check_images(dataset_path, verbose=False):
    """
    Check all PNG files in the dataset for corruption
    
    Args:
        dataset_path (str): Path to dataset directory
        verbose (bool): Print progress for every image checked
    
    Returns:
        tuple: (total_images, corrupted_images_list)
    """
    # Look for images in common dataset structures
    possible_paths = [
        os.path.join(dataset_path, 'images'),  # Standard structure
        os.path.join(dataset_path, 'input'),   # COLMAP structure
        dataset_path  # Direct path
    ]
    
    png_files = []
    for path in possible_paths:
        if os.path.exists(path):
            files = glob.glob(os.path.join(path, '**/*.png'), recursive=True)
            png_files.extend(files)
    
    if not png_files:
        print(f"❌ No PNG files found in {dataset_path}")
        print("   Checked paths:")
        for path in possible_paths:
            print(f"   - {path}")
        return 0, []
    
    print(f"🔍 Found {len(png_files)} PNG files in {dataset_path}")
    
    corrupted = []
    for i, img_path in enumerate(png_files):
        try:
            with Image.open(img_path) as img:
                img.verify()  # This will raise an exception if corrupted
                
            # Re-open to check if we can actually load the image data
            with Image.open(img_path) as img:
                img.load()  # Force loading of image data
                
            if verbose or (i % 100 == 0 and i > 0):
                print(f"✅ Checked {i+1}/{len(png_files)} images...")
                
        except Exception as e:
            corrupted.append((img_path, str(e)))
            print(f"❌ CORRUPTED: {img_path}")
            print(f"   Error: {e}")
    
    return len(png_files), corrupted

def suggest_fixes(corrupted_images):
    """Suggest fixes for corrupted images"""
    if not corrupted_images:
        return
    
    print(f"\n🔧 SUGGESTED FIXES for {len(corrupted_images)} corrupted images:")
    print("=" * 60)
    
    for img_path, error in corrupted_images:
        path_obj = Path(img_path)
        parent_dir = path_obj.parent
        filename = path_obj.stem
        extension = path_obj.suffix
        
        # Extract frame number if possible
        try:
            frame_num = int(''.join(filter(str.isdigit, filename)))
            prev_frame = f"{frame_num-1:04d}{extension}"
            next_frame = f"{frame_num+1:04d}{extension}"
            
            print(f"\n📁 {img_path}")
            print(f"   Try replacing with nearby frames:")
            print(f"   cp {parent_dir}/{prev_frame} {img_path}")
            print(f"   # or")
            print(f"   cp {parent_dir}/{next_frame} {img_path}")
            
        except ValueError:
            print(f"\n📁 {img_path}")
            print(f"   Find a similar image in {parent_dir}/ and copy it:")
            print(f"   cp {parent_dir}/similar_image.png {img_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Check dataset images for corruption",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_images.py                                    # Check default dataset
  python check_images.py datasets/cut_roasted_beef         # Check specific dataset
  python check_images.py datasets/vrig-chicken --verbose   # Verbose output
        """
    )
    
    parser.add_argument(
        'dataset_path', 
        nargs='?', 
        default='datasets/cut_roasted_beef',
        help='Path to dataset directory (default: datasets/cut_roasted_beef)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print progress for every image checked'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"❌ Dataset path does not exist: {args.dataset_path}")
        print("\nAvailable datasets:")
        datasets_dir = "datasets"
        if os.path.exists(datasets_dir):
            for item in os.listdir(datasets_dir):
                item_path = os.path.join(datasets_dir, item)
                if os.path.isdir(item_path):
                    print(f"  - {item_path}")
        sys.exit(1)
    
    print(f"🔍 Checking images in: {args.dataset_path}")
    print("-" * 50)
    
    total_images, corrupted = check_images(args.dataset_path, args.verbose)
    
    print("\n" + "=" * 50)
    if not corrupted:
        print("✅ ALL IMAGES ARE VALID!")
        print(f"   Successfully checked {total_images} images")
    else:
        print(f"❌ FOUND {len(corrupted)} CORRUPTED IMAGES out of {total_images}")
        suggest_fixes(corrupted)
        
        print(f"\n📋 SUMMARY:")
        print(f"   Total images: {total_images}")
        print(f"   Corrupted: {len(corrupted)}")
        print(f"   Success rate: {((total_images - len(corrupted)) / total_images * 100):.1f}%")
        
        print(f"\n🔄 After fixing, re-run: python check_images.py {args.dataset_path}")

if __name__ == "__main__":
    main() 