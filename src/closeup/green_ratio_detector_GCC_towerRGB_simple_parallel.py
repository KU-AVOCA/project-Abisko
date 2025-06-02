#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallelized Green Ratio Detector

This script processes images in parallel to quantify vegetation coverage using Green Chromatic Coordinate (GCC) index.
It extracts datetime from EXIF metadata and exports basic green metrics to CSV.

Shunan Feng (shf@ign.ku.dk)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import tqdm
import datetime
from PIL import Image
from PIL.ExifTags import TAGS
import seaborn as sns
import concurrent.futures
import pandas as pd
from functools import partial

sns.set_theme(style="darkgrid", font_scale=1.5)

# Input and output directories
imfolder = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/1_Simon/1_Abisko/6_Tower_Data/Tower RGB images/1 Data/1 Years'
imoutfolder = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/Tower RGB images/Data_greenessByShunan_simple_parallel'

# Find all image files
imfiles = []
for ext in ['*.JPG', '*.jpg', '*.JPEG', '*.jpeg', '*.png', '*.PNG']:
    imfiles.extend(glob.glob(os.path.join(imfolder, '**/', ext), recursive=True))
print(f"Found {len(imfiles)} images in {imfolder}")

# Create output directories
if not os.path.exists(imoutfolder):
    os.makedirs(imoutfolder)

results_dir = os.path.join(imoutfolder, 'results')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

def get_image_datetime(image_path):
    """Extract datetime from EXIF metadata."""
    try:
        with Image.open(image_path) as img:
            exifdata = img._getexif()
            
            if exifdata is None:
                return None
                
            # Look for DateTimeOriginal (36867) or DateTime (306) tag
            datetime_taken = None
            for tag_id, value in exifdata.items():
                tag = TAGS.get(tag_id, tag_id)
                
                if tag == 'DateTimeOriginal' or tag == 'DateTime':
                    datetime_taken = value
                    break
            
            if datetime_taken:
                try:
                    dt = datetime.datetime.strptime(datetime_taken, '%Y:%m:%d %H:%M:%S')
                    return dt.strftime('%Y-%m-%d %H:%M:%S')
                except ValueError:
                    return datetime_taken
                    
            return None
    except Exception as e:
        return None

def calculate_green_metrics(img):
    """Calculate green metrics using GCC."""
    try:
        # Split the image into its BGR channels
        b, g, r = cv2.split(img)
        
        # Convert to float
        b = b.astype(float)
        g = g.astype(float)
        r = r.astype(float)
        
        # Calculate the Greenness index G/(R+G+B)
        denominator = np.maximum(r + g + b, 1e-10)
        greenness = g / denominator
        
        # Calculate mean and std of greenness for ALL valid pixels
        mean_greenness = np.mean(greenness)
        std_greenness = np.std(greenness)
        
        # Create a binary mask using a threshold
        threshold = 0.38 + 0.01 
        green_mask = np.zeros_like(greenness, dtype=np.uint8)
        green_mask[greenness > threshold] = 255
        
        # Count green pixels and calculate ratio
        green_pixels = np.sum(green_mask > 0)
        total_pixels = img.shape[0] * img.shape[1]
        green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0
            
        green_metrics = {
            'ratio': green_ratio,
            'mean': mean_greenness,
            'std': std_greenness
        }
        
        return green_metrics, green_mask
        
    except Exception as e:
        return None, None

def process_single_image(img_path, output_base_dir):
    """Process a single image and return results."""
    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            return {
                'filename': img_path,
                'datetime': 'NA',
                'green_ratio': 'NA',
                'green_mean': 'NA', 
                'green_std': 'NA',
                'status': 'failed',
                'error': 'Could not read image'
            }
        
        # Get image datetime from EXIF metadata
        image_datetime = get_image_datetime(img_path)
        datetime_str = image_datetime if image_datetime else "NA"
        
        # Calculate green metrics
        green_metrics, green_mask = calculate_green_metrics(img)
        
        if green_metrics is None:
            return {
                'filename': img_path,
                'datetime': datetime_str,
                'green_ratio': 'NA',
                'green_mean': 'NA',
                'green_std': 'NA',
                'status': 'failed',
                'error': 'Green metrics calculation failed'
            }
        
        # Create masked image
        masked_img = cv2.bitwise_and(img, img, mask=(green_mask // 255).astype(np.uint8))
        
        # Convert images to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masked_img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
        
        # Create visualization figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Display original image
        axes[0].imshow(img_rgb)
        title_text = f"Original ({datetime_str})" if datetime_str != "NA" else "Original"
        axes[0].set_title(title_text)
        axes[0].axis('off')
        
        # Display masked image
        axes[1].imshow(masked_img_rgb)
        axes[1].set_title(f"Green Masked (Ratio={green_metrics['ratio']:.2f}, Mean={green_metrics['mean']:.2f})")
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Create output directory structure that mirrors input
        rel_path = os.path.relpath(os.path.dirname(img_path), imfolder)
        output_dir = os.path.join(output_base_dir, rel_path)
        
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract base filename and extension
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        extension = '.png'  # Use png for consistent output
        
        # Save the figure
        output_path = os.path.join(output_dir, f"{base_filename}_green_masked{extension}")
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)  # Close the figure to free memory
        
        # Return results for CSV
        return {
            'filename': img_path,
            'datetime': datetime_str,
            'green_ratio': green_metrics['ratio'],
            'green_mean': green_metrics['mean'],
            'green_std': green_metrics['std'],
            'status': 'success'
        }
    
    except Exception as e:
        return {
            'filename': img_path,
            'datetime': 'NA',
            'green_ratio': 'NA',
            'green_mean': 'NA',
            'green_std': 'NA',
            'status': 'failed',
            'error': str(e)
        }

def main():
    # Prepare for parallel processing
    # Determine optimal number of workers (adjust based on your system)
    max_workers = os.cpu_count() - 1  # Leave one CPU free for system operations
    max_workers = max(1, min(max_workers, 8))  # Cap at 8 workers to avoid memory issues
    
    # Create a partial function with the output directory
    process_func = partial(process_single_image, output_base_dir=imoutfolder)
    
    # Process images in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a progress bar
        futures = {executor.submit(process_func, img_path): img_path for img_path in imfiles}
        
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing images"):
            img_path = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                results.append({
                    'filename': img_path,
                    'datetime': 'NA',
                    'green_ratio': 'NA',
                    'green_mean': 'NA',
                    'green_std': 'NA',
                    'status': 'failed',
                    'error': str(e)
                })
    
    # Create DataFrame from results and save to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, 'green_ratio_simple.csv')
    df.to_csv(csv_path, columns=['filename', 'datetime', 'green_ratio', 'green_mean', 'green_std'], index=False)
    
    # Print summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    fail_count = sum(1 for r in results if r['status'] == 'failed')
    print(f"Done! Processed {len(results)} images: {success_count} successful, {fail_count} failed")
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()