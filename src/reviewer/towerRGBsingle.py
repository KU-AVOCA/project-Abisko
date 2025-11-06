#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single Image Green Ratio Detector with White Background

This script processes a single image to highlight vegetation using GCC index,
with non-green pixels converted to white instead of black.

Shunan Feng (shf@ign.ku.dk)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from PIL.ExifTags import TAGS
import datetime
import seaborn as sns

sns.set_theme(style="darkgrid", font_scale=1.5)

# ============ USER CONFIGURATION ============
# Specify the path to your image here
image_path = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/1_Simon/1_Abisko/6_Tower_Data/Tower RGB images/1 Data/1 Years/West-facing/2023/IMG_7920.JPG'

# Specify output directory
output_folder = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/qgis'
# ============================================

def get_image_datetime(image_path):
    """Extract datetime from EXIF metadata."""
    try:
        with Image.open(image_path) as img:
            exifdata = img._getexif()
            
            if exifdata is None:
                return None
                
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
        print(f"Error extracting metadata: {e}")
        return None

def calculate_green_metrics(img):
    """Calculate green metrics using Green Chromatic Coordinate (GCC)."""
    try:
        b, g, r = cv2.split(img)
        
        b = b.astype(float)
        g = g.astype(float)
        r = r.astype(float)
        
        denominator = np.maximum(r + g + b, 1e-10)
        greenness = g / denominator
        
        mean_greenness = np.mean(greenness)
        std_greenness = np.std(greenness)
        
        threshold = 0.38
        green_mask = np.zeros_like(greenness, dtype=np.uint8)
        green_mask[greenness > threshold] = 255
        
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

# Create output directory
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Check if image exists
if not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}")
    print("Please update the 'image_path' variable in the script.")
    exit(1)

# Load image
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not read image: {image_path}")
    exit(1)

print(f"Processing: {image_path}")

# Get datetime
image_datetime = get_image_datetime(image_path)
datetime_str = image_datetime if image_datetime else "NA"
print(f"Image datetime: {datetime_str}")

# Calculate green metrics
green_metrics, green_mask = calculate_green_metrics(img)

if green_metrics is not None:
    print(f"Green pixel ratio: {green_metrics['ratio']:.4f}")
    print(f"Green mean: {green_metrics['mean']:.4f}, std: {green_metrics['std']:.4f}")
    
    # Create white background image
    white_bg_img = np.ones_like(img) * 255  # Create white image
    
    # Copy green pixels from original image to white background
    mask_3channel = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
    white_bg_img = np.where(mask_3channel == 255, img, white_bg_img)
    
    # Convert images to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    white_bg_img_rgb = cv2.cvtColor(white_bg_img, cv2.COLOR_BGR2RGB)
    
    # Create visualization figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display original image
    axes[0].imshow(img_rgb)
    title_text = f"Original ({datetime_str})" if datetime_str != "NA" else "Original"
    axes[0].set_title(title_text)
    axes[0].axis('off')
    
    # Display white background image
    axes[1].imshow(white_bg_img_rgb)
    axes[1].set_title(f"Green Ratio = {green_metrics['ratio']:.2f}")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Generate output filename
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    extension = os.path.splitext(os.path.basename(image_path))[1]
    
    # Save visualization
    output_vis_path = os.path.join(output_folder, f"{base_filename}_green_white_bg_comparison.png")
    fig.savefig(output_vis_path, bbox_inches='tight', dpi=300)
    print(f"Saved comparison to: {output_vis_path}")
    
    # Save just the green-on-white image
    output_img_path = os.path.join(output_folder, f"{base_filename}_green_white_bg{extension}")
    cv2.imwrite(output_img_path, white_bg_img)
    print(f"Saved green-on-white image to: {output_img_path}")
    
    plt.show()
    
else:
    print(f"Vegetation quantification failed")

print("Done!")