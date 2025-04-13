#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Green Ratio Detector

This script processes images to quantify vegetation coverage using Green Chromatic Coordinate (GCC) index.
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

sns.set_theme(style="darkgrid", font_scale=1.5)

# Input and output directories
imfolder = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/1_Simon/1_Abisko/6_Tower_Data/Tower RGB images/1 Data/1 Years'
imoutfolder = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/Tower RGB images/Data_greenessByShunan_simple_mean'

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

# Initialize CSV file
csv_path = os.path.join(results_dir, 'green_ratio_simple.csv')
with open(csv_path, 'w') as f:
    f.write('filename,datetime,green_ratio,green_mean,green_std\n')

def get_image_datetime(image_path):
    """
    Extract the datetime when the image was taken from EXIF metadata.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        str: Datetime string in ISO format (YYYY-MM-DD HH:MM:SS) or None if not available
    """
    try:
        with Image.open(image_path) as img:
            # Extract EXIF data
            exifdata = img._getexif()
            
            if exifdata is None:
                return None
                
            # Look for DateTimeOriginal (36867) or DateTime (306) tag
            datetime_taken = None
            for tag_id, value in exifdata.items():
                tag = TAGS.get(tag_id, tag_id)
                
                # Check for different date time tags
                if tag == 'DateTimeOriginal' or tag == 'DateTime':
                    datetime_taken = value
                    break
            
            if datetime_taken:
                # Convert from EXIF format (YYYY:MM:DD HH:MM:SS) to ISO format
                try:
                    # Parse the datetime
                    dt = datetime.datetime.strptime(datetime_taken, '%Y:%m:%d %H:%M:%S')
                    return dt.strftime('%Y-%m-%d %H:%M:%S')
                except ValueError:
                    # In case the format is unexpected
                    return datetime_taken
                    
            return None
    except Exception as e:
        print(f"Error extracting metadata from {image_path}: {e}")
        return None

def calculate_green_metrics(img):
    """
    Calculate green metrics using Green Chromatic Coordinate (GCC).
    
    Args:
        img: Input image in BGR format
        
    Returns:
        tuple: green_metrics dictionary, green_mask
    """
    try:
        # Split the image into its BGR channels
        b, g, r = cv2.split(img)
        
        # Convert to float to avoid integer division
        b = b.astype(float)
        g = g.astype(float)
        r = r.astype(float)
        
        # Calculate the Greenness index G/(R+G+B)
        # Avoid division by zero using np.maximum to ensure denominator is never zero
        denominator = np.maximum(r + g + b, 1e-10)
        greenness = g / denominator
        
        # Calculate mean and std of greenness for ALL valid pixels
        mean_greenness = np.mean(greenness)
        std_greenness = np.std(greenness)
        
        # Create a binary mask using a threshold
        threshold = 0.38  # Adjust as needed
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
        print(f"An error occurred: {e}")
        return None, None

# Process each image
for i in tqdm.tqdm(imfiles, desc="Processing images"):
    img = cv2.imread(i)
    if img is None:
        print(f"Could not read image: {i}")
        continue
        
    print("processing: ", i)

    # Get image datetime from EXIF metadata
    image_datetime = get_image_datetime(i)
    datetime_str = image_datetime if image_datetime else "NA"
    print(f"Image datetime: {datetime_str}")
    
    # Calculate green metrics
    green_metrics, green_mask = calculate_green_metrics(img)

    if green_metrics is not None:
        print(f"The green pixel ratio is: {green_metrics['ratio']:.4f}")
        print(f"Green mean: {green_metrics['mean']:.4f}, std: {green_metrics['std']:.4f}")
        
        # Write results to CSV
        with open(csv_path, 'a') as f:
            f.write(f'{i},{datetime_str},'
                   f'{green_metrics["ratio"]},{green_metrics["mean"]},{green_metrics["std"]}\n')

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
        rel_path = os.path.relpath(os.path.dirname(i), imfolder)
        output_dir = os.path.join(imoutfolder, rel_path)
        
        # Create the directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Extract base filename and extension
        base_filename = os.path.splitext(os.path.basename(i))[0]
        extension = os.path.splitext(os.path.basename(i))[1]
        
        # Save the figure to the mirrored directory structure
        output_path = os.path.join(output_dir, f"{base_filename}_green_masked{extension}")
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)  # Close the figure to free memory

    else:
        print(f"Vegetation quantification failed for {i}")
        # Write failure to CSV
        with open(csv_path, 'a') as f:
            f.write(f'{i},{datetime_str},NA,NA,NA\n')

print(f"Done! Results saved to {csv_path}")
