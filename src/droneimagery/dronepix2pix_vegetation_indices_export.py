#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Vegetation Indices Script

This script reads multispectral imagery, calculates selected vegetation indices,
exports pixel values to CSV, and generates visualization maps.

Author: [Your Name]
"""
#%%
import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import cmocean
from rasterio.plot import plotting_extent
import cmocean
import seaborn as sns

sns.set_theme(style="darkgrid", font_scale=1.5)

#%%
def calculate_index(src, index_name):
    """
    Calculate a specific vegetation index from the rasterio source.
    
    Parameters:
        src: The rasterio dataset source
        index_name (str): Name of the index to calculate
        
    Returns:
        np.ndarray: Calculated index values
    """
    if index_name == 'NDVI':
        band_red = src.read(3)
        band_nir = src.read(5)
        return (band_nir - band_red) / (band_nir + band_red)
    
    elif index_name == 'EVI':
        band_red = src.read(3)
        band_nir = src.read(5)
        band_blue = src.read(1)
        return 2.5 * (band_nir - band_red) / (band_nir + 6 * band_red - 7.5 * band_blue + 1)
    
    elif index_name == 'GNDVI':
        band_green = src.read(2)
        band_nir = src.read(5)
        return (band_nir - band_green) / (band_nir + band_green)
    
    elif index_name == 'GLI':
        band_blue = src.read(1)
        band_green = src.read(2)
        band_red = src.read(3)
        return (2 * band_green - band_blue - band_red) / (2 * band_green + band_blue + band_red)
        
    elif index_name == 'GCC':
        band_blue = src.read(1)
        band_green = src.read(2)
        band_red = src.read(3)
        return band_green / (band_green + band_blue + band_red)
    
    # Single bands
    elif index_name == 'band1' or index_name == 'blue':
        return src.read(1)
    elif index_name == 'band2' or index_name == 'green':
        return src.read(2)
    elif index_name == 'band3' or index_name == 'red':
        return src.read(3)
    elif index_name == 'band4' or index_name == 'red_edge':
        return src.read(4)
    elif index_name == 'band5' or index_name == 'nir':
        return src.read(5)
    else:
        raise ValueError(f"Index '{index_name}' not supported.")

def create_index_map(index_array, index_name, transform, crs, extent, output_path, 
                    colormap=None, vmin=None, vmax=None):
    """
    Create and save a visualization map for an index
    
    Parameters:
        index_array: The calculated index array
        index_name: Name of the index
        transform: Spatial transform from the source image
        crs: Coordinate reference system
        extent: Plotting extent
        output_path: Path to save the map
        colormap: Colormap to use (string name or colormap object, default is cmocean.cm.algae)
        vmin: Minimum value for color scaling (default is data minimum)
        vmax: Maximum value for color scaling (default is data maximum)
        
    Returns:
        str: Path to saved map
    """
    # Set default colormap if not provided
    if colormap is None:
        colormap = cmocean.cm.algae
    elif isinstance(colormap, str):
        # If colormap is a string, try to get it from matplotlib or cmocean
        try:
            if hasattr(cmocean.cm, colormap):
                colormap = getattr(cmocean.cm, colormap)
            elif hasattr(plt.cm, colormap):
                colormap = getattr(plt.cm, colormap)
            else:
                print(f"Colormap '{colormap}' not found, using default.")
                colormap = cmocean.cm.algae
        except:
            print(f"Error loading colormap '{colormap}', using default.")
            colormap = cmocean.cm.algae
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Apply vmin/vmax if provided, otherwise use data range
    im = ax.imshow(index_array, cmap=colormap, extent=extent, vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar(im, ax=ax, orientation='vertical')
    cbar.set_label(index_name)
    
    # Add metadata
    metadata_text = (
        f"Resolution: {transform[0]:.2f} m\n"
        f"CRS: {crs.to_string()}\n"
    )
    ax.text(0.02, 0.02, metadata_text, transform=ax.transAxes, 
           verticalalignment='bottom', 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    
    return output_path

def export_vegetation_indices(image_path, index_names, output_dir=None, 
                             colormaps=None, value_ranges=None):
    """
    Calculate specified vegetation indices from a multispectral image and export:
    1. CSV file with pixel values
    2. Visualization maps
    
    Parameters:
        image_path (str): Path to input multispectral GeoTIFF
        index_names (list): List of index names to calculate (e.g., ['NDVI', 'GCC'])
        output_dir (str): Directory to save outputs (defaults to image directory)
        colormaps (dict): Dictionary mapping index names to colormaps
        value_ranges (dict): Dictionary mapping index names to (vmin, vmax) tuples
        
    Returns:
        tuple: Paths to the CSV file and map images
    """
    if len(index_names) != 2:
        raise ValueError("Please specify exactly two indices to calculate and compare")
    
    print(f"Processing {os.path.basename(image_path)} with indices: {', '.join(index_names)}...")
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Initialize default colormaps and value ranges if not provided
    if colormaps is None:
        colormaps = {}
    if value_ranges is None:
        value_ranges = {}
    
    # Open the image and calculate indices
    with rasterio.open(image_path) as src:
        # Calculate the requested indices
        indices = {}
        for index_name in index_names:
            indices[index_name] = calculate_index(src, index_name)
            
        # Get image metadata
        transform = src.transform
        crs = src.crs
        extent = plotting_extent(src)
    
    # Create mask for valid data (where both indices have values)
    valid_mask = True
    for index_name in index_names:
        valid_mask = valid_mask & (~np.isnan(indices[index_name]))
    
    # Prepare CSV data - extract values where both indices are valid
    index_values = {}
    for index_name in index_names:
        index_values[index_name] = indices[index_name][valid_mask]
    
    # Create DataFrame and export to CSV
    df = pd.DataFrame(index_values)
    
    index_names_str = "_".join(index_names)
    csv_path = os.path.join(output_dir, f"{base_name}_{index_names_str}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Pixel values exported to: {csv_path}")
    
    # Generate maps
    map_paths = []
    
    # Create maps for each index
    for index_name in index_names:
        map_path = os.path.join(output_dir, f"{base_name}_{index_name}_map.png")
        
        # Get colormap and value range for this index if specified
        colormap = colormaps.get(index_name, None)
        
        vmin = None
        vmax = None
        if index_name in value_ranges:
            vmin, vmax = value_ranges[index_name]
            
        create_index_map(
            indices[index_name], 
            index_name, 
            transform, 
            crs, 
            extent, 
            map_path,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax
        )
        map_paths.append(map_path)
        print(f"Map for {index_name} saved to: {map_path}")
    
    return csv_path, map_paths


def process_directory(input_dir, index_names, output_dir=None, pattern="*.tif", 
                     colormaps=None, value_ranges=None):
    """
    Process all tif files in a directory
    
    Parameters:
        input_dir (str): Directory containing GeoTIFF files
        index_names (list): List of index names to calculate (e.g., ['NDVI', 'GCC'])
        output_dir (str): Directory to save outputs
        pattern (str): File pattern to match
        colormaps (dict): Dictionary mapping index names to colormaps
        value_ranges (dict): Dictionary mapping index names to (vmin, vmax) tuples
    """
    import glob
    
    files = glob.glob(os.path.join(input_dir, pattern))
    if not files:
        print(f"No files found matching pattern '{pattern}' in {input_dir}")
        return
        
    print(f"Found {len(files)} files to process")
    for file_path in files:
        try:
            export_vegetation_indices(file_path, index_names, output_dir, 
                                     colormaps, value_ranges)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    # === CHOOSE YOUR INDICES HERE ===
    # Available indices: 'NDVI', 'EVI', 'GNDVI', 'GLI', 'GCC', 'blue', 'green', 'red', 'red_edge', 'nir'
    index_names = ['NDVI', 'GCC']  # Change these to any two indices you want to compare
    
    # === CUSTOMIZE VISUALIZATIONS HERE ===
    # Set specific colormaps for each index
    # Available colormaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    # or from cmocean: https://matplotlib.org/cmocean/
    colormaps = {
        'NDVI': 'balance',  # Use matplotlib's viridis colormap for NDVI
        'GCC': 'speed'  # Use cmocean's algae colormap for GCC
    }
    
    # Set specific value ranges for color scaling
    value_ranges = {
        'NDVI': (-1, 1),  # 
        'GCC': (0, 1)     # 
    }
    
    # Example usage:
    # For a single file:
    # export_vegetation_indices("/path/to/your/image.tif", index_names, 
    #                          colormaps=colormaps, value_ranges=value_ranges)
    
    # For a directory of files:
    # process_directory("/path/to/directory", index_names, "/path/to/output", 
    #                  colormaps=colormaps, value_ranges=value_ranges)
    
    # Replace with your actual file path
    image_path = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/3_Shunan/data/studentdebug/23_06_08_orthomosaic_georef.tif"
    export_vegetation_indices(image_path, index_names, 
                             colormaps=colormaps, value_ranges=value_ranges)