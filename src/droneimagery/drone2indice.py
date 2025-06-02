#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vegetation Index Calculator and GeoTIFF Exporter

This script reads multispectral imagery, calculates selected vegetation indices,
and exports them as GeoTIFF files.

Shunan Feng (shf@ign.ku.dk)
"""
#%%
import os
import glob
import numpy as np
import rasterio
from rasterio.dtypes import float32
import tqdm
import geopandas as gpd
from shapely.geometry import box

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
        band_red = src.read(3).astype(np.float32)
        band_nir = src.read(5).astype(np.float32)
        # Avoid division by zero
        denominator = band_nir + band_red
        result = np.full_like(denominator, np.nan)
        valid = denominator != 0
        result[valid] = (band_nir[valid] - band_red[valid]) / denominator[valid]
        return result
    
    elif index_name == 'EVI':
        band_red = src.read(3).astype(np.float32)
        band_nir = src.read(5).astype(np.float32)
        band_blue = src.read(1).astype(np.float32)
        # Avoid division by zero
        denominator = band_nir + 6 * band_red - 7.5 * band_blue + 1
        result = np.full_like(denominator, np.nan)
        valid = denominator != 0
        result[valid] = 2.5 * (band_nir[valid] - band_red[valid]) / denominator[valid]
        return result
    
    elif index_name == 'GNDVI':
        band_green = src.read(2).astype(np.float32)
        band_nir = src.read(5).astype(np.float32)
        # Avoid division by zero
        denominator = band_nir + band_green
        result = np.full_like(denominator, np.nan)
        valid = denominator != 0
        result[valid] = (band_nir[valid] - band_green[valid]) / denominator[valid]
        return result
    
    elif index_name == 'GLI':
        band_blue = src.read(1).astype(np.float32)
        band_green = src.read(2).astype(np.float32)
        band_red = src.read(3).astype(np.float32)
        # Avoid division by zero
        denominator = 2 * band_green + band_blue + band_red
        result = np.full_like(denominator, np.nan)
        valid = denominator != 0
        result[valid] = (2 * band_green[valid] - band_blue[valid] - band_red[valid]) / denominator[valid]
        return result
        
    elif index_name == 'GCC':
        band_blue = src.read(1).astype(np.float32)
        band_green = src.read(2).astype(np.float32)
        band_red = src.read(3).astype(np.float32)
        # Avoid division by zero
        denominator = band_green + band_blue + band_red
        result = np.full_like(denominator, np.nan)
        valid = denominator != 0
        result[valid] = band_green[valid] / denominator[valid]
        return result
    
    elif index_name == 'LAI': # Landsat and Sentinel-2 based
        # (b1 - b5) / [(b1 + b5) -2*b2]
        band_blue = src.read(1).astype(np.float32)
        band_green = src.read(2).astype(np.float32)
        band_nir = src.read(5).astype(np.float32)
        # Avoid division by zero
        denominator = band_blue + band_nir - 2 * band_green
        result = np.full_like(denominator, np.nan)
        valid = denominator != 0
        result[valid] = (band_blue[valid] - band_nir[valid]) / denominator[valid]
        return result
    
    # Single bands
    elif index_name == 'band1' or index_name == 'blue':
        return src.read(1).astype(np.float32)
    elif index_name == 'band2' or index_name == 'green':
        return src.read(2).astype(np.float32)
    elif index_name == 'band3' or index_name == 'red':
        return src.read(3).astype(np.float32)
    elif index_name == 'band4' or index_name == 'red_edge':
        return src.read(4).astype(np.float32)
    elif index_name == 'band5' or index_name == 'nir':
        return src.read(5).astype(np.float32)
    else:
        raise ValueError(f"Index '{index_name}' not supported.")

def save_index_as_geotiff(index_array, src_image, output_path):
    """
    Save a calculated index as a GeoTIFF file.
    
    Parameters:
        index_array (numpy.ndarray): The calculated index array
        src_image (rasterio.DatasetReader): Source image to get metadata from
        output_path (str): Path to save the output GeoTIFF
        
    Returns:
        str: Path to the saved GeoTIFF file
    """
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a new GeoTIFF with metadata from source file
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=src_image.height,
        width=src_image.width,
        count=1,
        dtype=float32,
        crs=src_image.crs,
        transform=src_image.transform,
        nodata=np.nan,
        BIGTIFF='IF_NEEDED'
    ) as dst:
        dst.write(index_array, 1)
    
    return output_path

def extract_extent_as_shapefile(image_path, output_path=None):
    """
    Extract the extent of a GeoTIFF file and save it as a shapefile.
    
    Parameters:
        image_path (str): Path to input GeoTIFF file
        output_path (str): Path to save output shapefile (default is same location as image)
        
    Returns:
        str: Path to the created shapefile
    """
    # Set default output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.dirname(image_path)
        output_path = os.path.join(output_dir, f"{base_name}_extent.shp")
    
    # Make sure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract raster bounds
    with rasterio.open(image_path) as src:
        # Get the bounds in the CRS of the raster
        bounds = src.bounds
        # Get the CRS of the raster
        crs = src.crs
    
    # Create a Shapely box from the bounds
    polygon = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    
    # Create a GeoDataFrame with a single polygon
    gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs=crs)
    
    # Save to shapefile
    gdf.to_file(output_path)
    
    print(f"Extent shapefile saved to: {output_path}")
    return output_path

def calculate_and_save_index(image_path, index_name, output_dir=None, create_extent=False):
    """
    Calculate a vegetation index from a multispectral image and save as GeoTIFF.
    
    Parameters:
        image_path (str): Path to input multispectral GeoTIFF
        index_name (str): Name of index to calculate (e.g., 'NDVI', 'GCC')
        output_dir (str): Directory to save output (defaults to image directory)
        create_extent (bool): Whether to create a shapefile of the image extent
        
    Returns:
        tuple: Paths to the output GeoTIFF file and shapefile (if requested)
    """
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Set output file path
    output_path = os.path.join(output_dir, f"{base_name}_{index_name}.tif")
    
    # Calculate index and save as GeoTIFF
    with rasterio.open(image_path) as src:
        index_array = calculate_index(src, index_name)
        save_index_as_geotiff(index_array, src, output_path)
    
    print(f"Index {index_name} saved to: {output_path}")
    
    # Create extent shapefile if requested
    if create_extent:
        extent_path = os.path.join(output_dir, f"{base_name}_extent.shp")
        extract_extent_as_shapefile(image_path, extent_path)
        return output_path, extent_path
    
    return output_path

def process_directory(input_dir, index_name, output_dir=None, pattern="*.tif", create_extent=False):
    """
    Process all tif files in a directory
    
    Parameters:
        input_dir (str): Directory containing GeoTIFF files
        index_name (str): Name of index to calculate
        output_dir (str): Directory to save outputs
        pattern (str): File pattern to match
        create_extent (bool): Whether to create shapefiles of the image extents
    """
    files = glob.glob(os.path.join(input_dir, pattern))
    if not files:
        print(f"No files found matching pattern '{pattern}' in {input_dir}")
        return
        
    print(f"Found {len(files)} files to process")
    for file_path in tqdm.tqdm(files, desc="Processing files", unit="file"):
        try:
            calculate_and_save_index(file_path, index_name, output_dir, create_extent)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

#%%
if __name__ == "__main__":
    # === CONFIGURE YOUR INDEX HERE ===
    # Available indices: 'NDVI', 'EVI', 'GNDVI', 'GLI', 'GCC', 'LAI', 
    #                    'blue', 'green', 'red', 'red_edge', 'nir'
    index_name = 'GCC'  # Change this to the index you want to calculate
    
    # Set to True if you want to create shapefiles of the image extents
    create_extent = True
    
    input_dir = "/media/sfm/NTFS2TB/Monika_Katrine_Abisko/orthomosaics_georef/21_06_29_orthomosaic_georef_bandmath/21_06_29_orthomosaic_georef_processed.tif"
    output_dir = "/media/sfm/Kobbefjord/Shunan/indices/"
    # Example usage for a single file:
    calculate_and_save_index(input_dir, index_name, output_dir, create_extent=create_extent)
    
    # Example usage for a directory of files:
    # input_dir = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/drone/orthomosaic4analysis/"
    # output_dir = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/drone/indices/"
    # process_directory(
    #     input_dir=input_dir, 
    #     index_name=index_name, 
    #     output_dir=output_dir,
    #     pattern="*georef_processed.tif",  # Change to match your files
    #     create_extent=create_extent
    # )