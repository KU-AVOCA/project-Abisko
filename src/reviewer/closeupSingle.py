#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single Image Green Ratio Detector with White Background (GCC Method)

This script processes a single image to quantify vegetation coverage using 
the Greenness Chromatic Coordinate (GCC) index, with non-green pixels 
converted to white background for publication purposes.

Shunan Feng (shf@ign.ku.dk)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

sns.set_theme(style="darkgrid", font_scale=1.5)

# ============ USER CONFIGURATION ============
# Specify the path to your image here
image_path = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/1_Simon/1_Abisko/4_CHMB_RGB/3_2023/2_Work/2_Cropped/8_05-08-2023/05-08-2023_E1.JPG"

# Specify output directory
output_folder = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/qgis'

# GCC threshold (adjust if needed, default is 0.38)
GCC_THRESHOLD = 0.38
# ============================================

def quantify_vegetation(img, threshold=0.38):
    """
    Quantifies the ratio of green to non-green pixels using Greenness index (GCC).

    Args:
        img: The input image (BGR format)
        threshold: GCC threshold value (default 0.38)

    Returns:
        tuple: Ratio of green pixels to total pixels, green mask, and mean greenness
    """
    try:
        # Split the image into its BGR channels
        b, g, r = cv2.split(img)
        
        # Convert to float to avoid integer division
        b = b.astype(float)
        g = g.astype(float)
        r = r.astype(float)
        
        # Calculate the Greenness index G/(R+G+B)
        greenness = g / (r + g + b + 1e-10)
        
        # Calculate mean greenness
        mean_greenness = np.mean(greenness)
        
        # Create a binary mask using threshold
        green_mask = (greenness > threshold).astype(np.uint8) * 255
        
        # Count green pixels and calculate ratio
        green_pixels = np.sum(green_mask > 0)
        total_pixels = img.shape[0] * img.shape[1]
        green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0
        
        return green_ratio, green_mask, mean_greenness

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None

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

# Quantify vegetation
ratio, green_mask, mean_greenness = quantify_vegetation(img, threshold=GCC_THRESHOLD)

if ratio is not None:
    print(f"Green pixel ratio: {ratio:.4f}")
    print(f"Mean GCC value: {mean_greenness:.4f}")
    print(f"Threshold used: {GCC_THRESHOLD}")
    
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
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Display white background image
    axes[1].imshow(white_bg_img_rgb)
    axes[1].set_title(f"Green Ratio = {ratio:.2f}")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Generate output filenames
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    extension = os.path.splitext(os.path.basename(image_path))[1]
    
    # Save comparison visualization
    output_vis_path = os.path.join(output_folder, f"{base_filename}_green_white_comparison.png")
    fig.savefig(output_vis_path, bbox_inches='tight', dpi=300)
    print(f"Saved comparison to: {output_vis_path}")
    
    # Save just the green-on-white image
    output_img_path = os.path.join(output_folder, f"{base_filename}_green_white_bg{extension}")
    cv2.imwrite(output_img_path, white_bg_img)
    print(f"Saved green-on-white image to: {output_img_path}")
    
    # Optionally show the plot
    plt.show()
    
else:
    print("Vegetation quantification failed.")

print("Done!")