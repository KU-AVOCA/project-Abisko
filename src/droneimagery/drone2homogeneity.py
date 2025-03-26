#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drone Image Homogeneity Analyzer

This script analyzes the spatial homogeneity characteristics of drone imagery
across different resolutions. It calculates various metrics to quantify how 
spatial patterns change as resolution changes.

The script can process:
1. A single GeoTIFF file to analyze its homogeneity
2. A directory of resampled GeoTIFFs to compare homogeneity across resolutions
3. A list of specific files to compare

Author: Shunan Feng (shf@ign.ku.dk)
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from scipy.stats import entropy
from tqdm import tqdm

from skimage.feature import graycomatrix, graycoprops
from skimage.measure import find_contours

import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from rasterio.plot import plotting_extent
import cmocean
import seaborn as sns

# Set plotting style
sns.set_theme(style="whitegrid", font_scale=1.5)

#%%

def calculate_index(image_path, index_name, imoutput_dir=None):
    """
    Calculate a specific index from input tif image.
    ALTUM sensor has 5 bands:
    band1 blue: 475nm (32nm bandwidth)
    band2 green: 560nm (27nm bandwidth)
    band3 red: 668nm (14nm bandwidth)
    band4 red edge: 717nm (12nm bandwidth)
    band5 nir: 842nm (57nm bandwidth)

    Index reference:
    https://www.indexdatabase.de/

    Parameters:
        image_path (str): Path to GeoTIFF file
        index_name (str): Name of the index to calculate
        
    Returns:
        np.ndarray: Calculated index values
    """

    if imoutput_dir is None:
        imoutput_dir = os.path.dirname(image_path)
    
    if index_name == 'NDVI':# Normalized Difference Vegetation Index
        with rasterio.open(image_path) as src:

            band_red = src.read(3)
            band_nir = src.read(5)
            index = (band_nir - band_red) / (band_nir + band_red)

    elif index_name == 'EVI':# Enhanced Vegetation Index
        with rasterio.open(image_path) as src:

            band_red = src.read(3)
            band_nir = src.read(5)
            band_blue = src.read(1)
            index = 2.5 * (band_nir - band_red) / (band_nir + 6 * band_red - 7.5 * band_blue + 1)
    
    elif index_name == 'GNDVI':# Green Normalized Difference Vegetation Index
        with rasterio.open(image_path) as src:

            band_green = src.read(2)
            band_nir = src.read(5)
            index = (band_nir - band_green) / (band_nir + band_green)

    elif index_name == 'SLAVI':# Specific Leaf Area Vegetation Index
        with rasterio.open(image_path) as src:
            
            band_red = src.read(3)
            band_nir = src.read(5)
            index = band_nir / (band_nir + band_red)

    elif index_name == 'GLI':# Green Leaf Index
        with rasterio.open(image_path) as src:

            band_blue = src.read(1)
            band_green = src.read(2)
            band_red = src.read(3)

            index = (2 * band_green - band_blue - band_red) / (2 * band_green + band_blue + band_red)
        
    elif index_name == 'GCC':# Green Chromatic Coordinate
        with rasterio.open(image_path) as src:

            band_blue = src.read(1)
            band_green = src.read(2)
            band_red = src.read(3)

            index = band_green / (band_green + band_blue + band_red)
        
    else:
        raise ValueError(f"Index '{index_name}' not supported.")
    
    # visualize the index
    with rasterio.open(image_path) as src:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # ax.set_title(f'{index_name} Index')
        extent = plotting_extent(src)
        plt.imshow(index, cmap=cmocean.cm.balance, extent=extent, ax=ax)
        scalebar = ScaleBar(
            src.transform[0], 
            location='lower right', 
            units='m', 
            dimension='si-length',
            scale_loc='bottom', 
            length_fraction=0.1
        )
        ax.add_artist(scalebar)
        
        # Add metadata as text
        metadata_text = (
            f"Index: {index_name}\n"
            f"Resolution: {src.transform[0]:.2f} m\n"
            f"CRS: {src.crs.to_string()}\n"
        )
        ax.text(0.02, 0.02, metadata_text, transform=ax.transAxes, 
                verticalalignment='bottom', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Tight layout
        plt.tight_layout()

        # Save the plot
        output_dir = os.path.dirname(image_path)
        output_path_png = os.path.join(output_dir, f"{os.path.basename(image_path).replace('.tif', '')}_{index_name}.png")

    
    return index

def analyze_homogeneity(image_path, band_indices='NDVI', output_dir=None):
    """
    Analyze the spatial homogeneity of a single image.
    
    Parameters:
        image_path (str): Path to GeoTIFF file
        band_indices (list): List of band indices to analyze (default=NDVI)
        output_dir (str): Directory to save results (default=same as image)
        
    Returns:
        pd.DataFrame: DataFrame with homogeneity metrics
    """
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare results storage
    band_results = []
    
    with rasterio.open(image_path) as src:
        # Determine bands to analyze
        if band_indices is None:
            band_indices = list(range(1, src.count + 1))
        
        # Get resolution and other metadata
        resolution = src.transform[0]
        image_name = os.path.basename(image_path)
        
        print(f"Analyzing {image_name} with resolution {resolution}m...")
        print(f"Processing {len(band_indices)} bands...")
        
        # Process each band
        for band_idx in band_indices:
            print(f"  Band {band_idx}...")
            
            # Read band data
            data = src.read(band_idx)
            
            # Check for NaN values
            mask = ~np.isnan(data)
            if not np.any(mask):
                print(f"  Band {band_idx} contains only NaN values. Skipping.")
                continue
                
            # Get valid data
            valid_data = data[mask]
            
            # Basic statistics
            mean_val = np.mean(valid_data)
            std_val = np.std(valid_data)
            cv = std_val / mean_val if mean_val != 0 else np.nan
            
            # Prepare result dictionary
            result = {
                'image': image_name,
                'resolution': resolution,
                'band': band_idx,
                'mean': mean_val,
                'std': std_val,
                'cv': cv,
                'min': np.min(valid_data),
                'max': np.max(valid_data)
            }
            
            # Calculate entropy
            if valid_data.size > 0:
                hist, _ = np.histogram(valid_data, bins=50, density=True)
                hist = hist[hist > 0]  # Remove zeros
                result['entropy'] = entropy(hist)
            else:
                result['entropy'] = np.nan
            
            # Calculate Moran's I (spatial autocorrelation)
            try:
                # Create weights matrix (Queen's case)
                w = ndimage.generate_binary_structure(2, 2)
                w[1, 1] = False  # Remove self
                
                # Handle NaNs for convolution
                data_centered = np.where(mask, data - np.mean(valid_data), 0)
                
                # Spatial lag
                spatial_lag = ndimage.convolve(data_centered, w, mode='constant', cval=0)
                
                # Calculate Moran's I
                numer = np.sum(data_centered * spatial_lag)
                denom = np.sum(data_centered * data_centered)
                
                if denom > 0:
                    n = np.sum(mask)  # Valid pixels
                    w_sum = np.sum(w)  # Sum of weights
                    moran_i = (n / w_sum) * (numer / denom)
                    result['moran_i'] = moran_i
                else:
                    result['moran_i'] = np.nan
            except Exception as e:
                print(f"  Error calculating Moran's I: {e}")
                result['moran_i'] = np.nan
            
            # Texture analysis with GLCM (if skimage is available)
            if HAVE_SKIMAGE:
                try:
                    # Normalize data to 0-255 range for GLCM
                    if np.max(valid_data) > np.min(valid_data):
                        norm_data = np.zeros_like(data)
                        norm_data[mask] = ((valid_data - np.min(valid_data)) / 
                                        (np.max(valid_data) - np.min(valid_data)) * 255).astype(np.uint8)
                        
                        # Take a sample if image is large (for performance)
                        if norm_data.size > 1000000:
                            # Find a central region with data
                            rows, cols = np.where(mask)
                            if len(rows) > 0:
                                center_row = int(np.mean(rows))
                                center_col = int(np.mean(cols))
                                
                                # Extract window
                                size = min(500, min(data.shape))
                                half_size = size // 2
                                r_start = max(0, center_row - half_size)
                                c_start = max(0, center_col - half_size)
                                r_end = min(data.shape[0], r_start + size)
                                c_end = min(data.shape[1], c_start + size)
                                
                                sample = norm_data[r_start:r_end, c_start:c_end]
                            else:
                                sample = norm_data
                        else:
                            sample = norm_data
                        
                        # Calculate GLCM
                        if np.sum(sample > 0) > 100:  # Ensure enough non-zero pixels
                            sample = np.nan_to_num(sample).astype(np.uint8)
                            glcm = graycomatrix(sample, 
                                               distances=[1], 
                                               angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                                               levels=256,
                                               symmetric=True, 
                                               normed=True)
                            
                            # Extract GLCM properties
                            result['glcm_contrast'] = graycoprops(glcm, 'contrast')[0, 0]
                            result['glcm_dissimilarity'] = graycoprops(glcm, 'dissimilarity')[0, 0]
                            result['glcm_homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
                            result['glcm_energy'] = graycoprops(glcm, 'energy')[0, 0]
                            result['glcm_correlation'] = graycoprops(glcm, 'correlation')[0, 0]
                            result['glcm_asm'] = graycoprops(glcm, 'ASM')[0, 0]
                        else:
                            result['glcm_contrast'] = np.nan
                            result['glcm_dissimilarity'] = np.nan
                            result['glcm_homogeneity'] = np.nan
                            result['glcm_energy'] = np.nan
                            result['glcm_correlation'] = np.nan
                            result['glcm_asm'] = np.nan
                    else:
                        # All pixels have the same value
                        result['glcm_contrast'] = 0
                        result['glcm_dissimilarity'] = 0
                        result['glcm_homogeneity'] = 1
                        result['glcm_energy'] = 1
                        result['glcm_correlation'] = np.nan
                        result['glcm_asm'] = 1
                except Exception as e:
                    print(f"  Error in GLCM calculation: {e}")
                    result['glcm_contrast'] = np.nan
                    result['glcm_dissimilarity'] = np.nan
                    result['glcm_homogeneity'] = np.nan
                    result['glcm_energy'] = np.nan
                    result['glcm_correlation'] = np.nan
                    result['glcm_asm'] = np.nan
            
            # Add to results
            band_results.append(result)
    
    # Create DataFrame from results
    if band_results:
        df = pd.DataFrame(band_results)
        
        # Save to CSV
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        csv_path = os.path.join(output_dir, f"{base_name}_homogeneity.csv")
        df.to_csv(csv_path, index=False)
        print(f"Homogeneity metrics saved to: {csv_path}")
        
        return df
    else:
        print("No valid bands found for analysis.")
        return None

def analyze_directory(directory, pattern="*.tif", output_dir=None, band_indices=None):
    """
    Analyze homogeneity for all images in a directory.
    
    Parameters:
        directory (str): Directory containing GeoTIFF files
        pattern (str): File pattern to match
        output_dir (str): Directory to save results
        band_indices (list): List of band indices to analyze
    
    Returns:
        pd.DataFrame: Combined DataFrame with all results
    """
    if output_dir is None:
        output_dir = os.path.join(directory, "homogeneity_analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all matching files
    files = glob.glob(os.path.join(directory, pattern))
    
    if not files:
        print(f"No files found matching pattern '{pattern}' in {directory}")
        return None
    
    print(f"Found {len(files)} files to analyze")
    
    # Process each file
    all_results = []
    for file_path in tqdm(files, desc="Analyzing files"):
        try:
            df = analyze_homogeneity(file_path, band_indices, output_dir)
            if df is not None:
                all_results.append(df)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Combine results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        combined_csv = os.path.join(output_dir, "combined_homogeneity.csv")
        combined_df.to_csv(combined_csv, index=False)
        print(f"Combined results saved to: {combined_csv}")
        
        # Create visualizations
        create_homogeneity_plots(combined_df, output_dir)
        
        return combined_df
    else:
        print("No valid results to combine.")
        return None

def create_homogeneity_plots(df, output_dir):
    """
    Create visualizations of homogeneity metrics across resolutions.
    
    Parameters:
        df (pd.DataFrame): DataFrame with homogeneity results
        output_dir (str): Directory to save plots
    """
    print("Creating homogeneity visualization plots...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have resolution data to plot
    if 'resolution' not in df.columns or df['resolution'].nunique() <= 1:
        print("Not enough resolution data for comparative plots.")
        return

    # Sort by resolution for proper plotting
    df = df.sort_values('resolution')
    
    # Determine which metrics to plot
    all_metrics = ['cv', 'entropy', 'moran_i']
    if 'glcm_homogeneity' in df.columns:
        all_metrics.extend(['glcm_homogeneity', 'glcm_contrast', 'glcm_energy', 'glcm_correlation'])
    
    available_metrics = [m for m in all_metrics if m in df.columns]
    
    if not available_metrics:
        print("No suitable metrics found for plotting.")
        return
    
    # Create plots for each band
    bands = df['band'].unique()
    
    for band in bands:
        band_df = df[df['band'] == band]
        
        # Plot each metric vs resolution
        for metric in available_metrics:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=band_df, x='resolution', y=metric, marker='o')
            plt.xscale('log')
            plt.title(f'Band {band}: {metric.replace("_", " ").title()} vs Resolution')
            plt.xlabel('Resolution (meters)')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'band{band}_{metric}_vs_resolution.png'), dpi=300)
            plt.close()
    
    # Create summary multi-metric plot for each band
    for band in bands:
        band_df = df[df['band'] == band]
        
        # Select up to 4 metrics for summary plot
        plot_metrics = available_metrics[:min(4, len(available_metrics))]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(plot_metrics):
            if i < len(axes):
                sns.lineplot(data=band_df, x='resolution', y=metric, marker='o', ax=axes[i])
                axes[i].set_xscale('log')
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_xlabel('Resolution (meters)')
                axes[i].set_ylabel(metric.replace('_', ' ').title())
                axes[i].grid(True, which='both', linestyle='--', alpha=0.7)
        
        # Hide any unused axes
        for i in range(len(plot_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Band {band}: Homogeneity Metrics vs Resolution', fontsize=16)
        plt.tight_layout()
        fig.subplots_adjust(top=0.92)
        plt.savefig(os.path.join(output_dir, f'band{band}_summary.png'), dpi=300)
        plt.close()
    
    # Create composite plot with all bands for selected metrics
    for metric in ['cv', 'entropy', 'moran_i']:
        if metric in available_metrics:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df, x='resolution', y=metric, hue='band', marker='o')
            plt.xscale('log')
            plt.title(f'{metric.replace("_", " ").title()} vs Resolution (All Bands)')
            plt.xlabel('Resolution (meters)')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'all_bands_{metric}.png'), dpi=300)
            plt.close()
    
    print(f"Visualization plots saved to: {output_dir}")

def main():
    """Main function to parse arguments and execute analysis"""
    parser = argparse.ArgumentParser(description='Analyze spatial homogeneity in drone imagery.')
    parser.add_argument('input', help='Input image file or directory')
    parser.add_argument('-o', '--output', help='Output directory for results')
    parser.add_argument('-b', '--bands', type=int, nargs='+', help='Band indices to analyze (default: all)')
    parser.add_argument('-p', '--pattern', default='*.tif', help='File pattern if input is a directory')
    parser.add_argument('-s', '--single', action='store_true', help='Process input as a single file, even if it is a directory')
    
    args = parser.parse_args()
    
    if args.single or not os.path.isdir(args.input):
        # Process single file
        analyze_homogeneity(args.input, args.bands, args.output)
    else:
        # Process directory
        analyze_directory(args.input, args.pattern, args.output, args.bands)

if __name__ == "__main__":
    main()