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
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from scipy.stats import entropy
from tqdm import tqdm

from skimage.feature import graycomatrix, graycoprops
# from skimage.measure import find_contours

import matplotlib.pyplot as plt
# from matplotlib_scalebar.scalebar import ScaleBar
from rasterio.plot import plotting_extent
import cmocean
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops

# Set plotting style
sns.set_theme(style="whitegrid", font_scale=1.5)

#%%

def func_calculate_index(image_path, index_name, imoutput_dir=None):
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
        imoutput_dir = os.path.join(os.path.dirname(image_path), index_name)
        os.makedirs(imoutput_dir, exist_ok=True)
    
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

    # elif index_name == 'SLAVI':# Specific Leaf Area Vegetation Index
    #     with rasterio.open(image_path) as src:
            
    #         band_red = src.read(3)
    #         band_nir = src.read(5)
    #         index = band_nir / (band_nir + band_red)

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

    # for single bands:
    elif index_name == 'band1' or index_name == 'blue':
        with rasterio.open(image_path) as src:
            index = src.read(1)
    elif index_name == 'band2' or index_name == 'green':
        with rasterio.open(image_path) as src:
            index = src.read(2)
    elif index_name == 'band3' or index_name == 'red':
        with rasterio.open(image_path) as src:
            index = src.read(3)
    elif index_name == 'band4' or index_name == 'red_edge':
        with rasterio.open(image_path) as src:
            index = src.read(4)
    elif index_name == 'band5' or index_name == 'nir':
        with rasterio.open(image_path) as src:
            index = src.read(5)
        
    else:
        raise ValueError(f"Index '{index_name}' not supported.")
    
    # visualize the index
    with rasterio.open(image_path) as src:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # ax.set_title(f'{index_name} Index')
        extent = plotting_extent(src)
        ax.imshow(index, cmap=cmocean.cm.algae, extent=extent)
        # scalebar = ScaleBar(
        #     src.transform[0], 
        #     location='lower right', 
        #     units='m', 
        #     dimension='si-length',
        #     scale_loc='bottom', 
        #     length_fraction=0.1
        # )
        # ax.add_artist(scalebar)
        # Add colorbar
        cbar = plt.colorbar(ax.images[0], ax=ax, orientation='vertical', 
                           fraction=0.046, pad=0.04)
        cbar.set_label(f'{index_name}')
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
        output_path_png = os.path.join(imoutput_dir, f"{os.path.basename(image_path).replace('.tif', '')}_{index_name}.png")
        fig.savefig(output_path_png, dpi=300)
        output_path_pdf = os.path.join(imoutput_dir, f"{os.path.basename(image_path).replace('.tif', '')}_{index_name}.pdf")
        fig.savefig(output_path_pdf, dpi=300)
        plt.close()
        print(f"Index visualization (png) saved to: {output_path_png}")
        print(f"Index visualization (pdf) saved to: {output_path_pdf}")

    
    return index



def calculate_and_analyze_index_directly(image_path, index_name, output_dir=None):
    """
    Calculate a vegetation index and directly analyze its spatial homogeneity without saving as GeoTIFF.
    
    Parameters:
        image_path (str): Path to input drone GeoTIFF
        index_name (str): Name of the index to calculate ('NDVI', 'EVI', etc.)
        output_dir (str): Directory to save analysis results
        
    Returns:
        pd.DataFrame: DataFrame with homogeneity metrics
    """
    print(f"Calculating and analyzing {index_name} for {os.path.basename(image_path)}...")
    
    # Create output directory based on index name
    if output_dir is None:
        parent_dir = os.path.dirname(image_path)
        output_dir = os.path.join(parent_dir, index_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate the vegetation index
    index_array = func_calculate_index(image_path, index_name, output_dir)
    
    # Prepare results storage
    result = {}
    
    with rasterio.open(image_path) as src:
        # Get resolution and other metadata
        resolution = src.transform[0]
        image_name = os.path.basename(image_path)
        
        # Create mask for valid data
        mask = ~np.isnan(index_array)
        if not np.any(mask):
            print(f"Warning: {index_name} contains only NaN values. Skipping homogeneity analysis.")
            return None
        
        # Get valid data
        valid_data = index_array[mask]
        
        # Basic statistics
        mean_val = np.mean(valid_data)
        std_val = np.std(valid_data)
        cv = std_val / mean_val if mean_val != 0 else np.nan
        
        # Prepare result dictionary
        result = {
            'image': image_name,
            'resolution': resolution,
            'band': index_name,  # Use index name as band identifier
            'index': index_name,
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
            data_centered = np.where(mask, index_array - np.mean(valid_data), 0)
            
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
        
        # Calculate GLCM texture metrics
        try:
            # Normalize data to 0-255 range for GLCM
            if np.max(valid_data) > np.min(valid_data):
                norm_data = np.zeros_like(index_array)
                norm_data[mask] = ((valid_data - np.min(valid_data)) / 
                                (np.max(valid_data) - np.min(valid_data)) * 255).astype(np.uint8)
                
                # Take a sample if image is large
                if norm_data.size > 1000000:
                    # Find a central region with data
                    rows, cols = np.where(mask)
                    if len(rows) > 0:
                        center_row = int(np.mean(rows))
                        center_col = int(np.mean(cols))
                        
                        # Extract window
                        size = min(500, min(index_array.shape))
                        half_size = size // 2
                        r_start = max(0, center_row - half_size)
                        c_start = max(0, center_col - half_size)
                        r_end = min(index_array.shape[0], r_start + size)
                        c_end = min(index_array.shape[1], c_start + size)
                        
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
    
    # Create DataFrame and save results
    df = pd.DataFrame([result])
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    csv_path = os.path.join(output_dir, f"{base_name}_{index_name}_homogeneity.csv")
    df.to_csv(csv_path, index=False)
    print(f"Homogeneity metrics saved to: {csv_path}")
    
    return df

#%%
def process_drone_images(input_path, indices=None, output_dir=None, pattern="*.tif"):
    """
    Process drone images without command line arguments.
    
    Parameters:
        input_path (str): Path to input image file or directory
        indices (list): List of indices to calculate ['NDVI', 'EVI', etc.]
        output_dir (str): Directory to save results
        resampled_dir (str): Directory with resampled images for multi-resolution analysis
        pattern (str): File pattern if input_path is a directory
    """
    if indices is None:
        indices = ['NDVI']
    
    print(f"Processing images with indices: {', '.join(indices)}")
    
    if os.path.isdir(input_path):
        # Process all files in directory
        files = glob.glob(os.path.join(input_path, pattern))
        if not files:
            print(f"No files found matching pattern '{pattern}' in {input_path}")
            return
            
        print(f"Found {len(files)} files to process")
        for file_path in tqdm(files, desc="Processing files"):
            for index_name in indices:
                try:
                    calculate_and_analyze_index_directly(file_path, index_name, output_dir)
                except Exception as e:
                    print(f"Error processing {file_path} with {index_name}: {e}")
    else:
        # Process single file with all requested indices
        print(f"Processing single file: {input_path}")
        for index_name in indices:
            calculate_and_analyze_index_directly(input_path, index_name, output_dir)

# Example usage
if __name__ == "__main__":
    # === EDIT THESE PARAMETERS FOR YOUR ANALYSIS ===
    
    # Option 1: Single image analysis - calculate and analyze NDVI for a single image
    # process_drone_images(
    #     input_path="/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/3_Shunan/data/studentdebug/23_06_08_orthomosaic_georef_processed_resampled_nearest/23_06_08_orthomosaic_georef_processed_0.5m_nearest.tif",
    #     indices=["NDVI"]
    # )
    
    # Option 2: Multi-resolution analysis
    # process_drone_images(
    #     input_path="/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/3_Shunan/data/studentdebug/23_06_08_orthomosaic_georef_processed.tif",
    #     resampled_dir="/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/3_Shunan/data/studentdebug/resampled",
    #     indices=["NDVI", "EVI", "GNDVI"],
    #     output_dir="/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/3_Shunan/data/studentdebug/results"
    # )
    
    # Option 3: Process all images in a directory
    process_drone_images(
        input_path="/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/3_Shunan/data/studentdebug/23_06_08_orthomosaic_georef_processed_resampled_nearest",
        indices=["NDVI", "GCC", "GLI", "SLAVI"],
        pattern="*.tif"
    )