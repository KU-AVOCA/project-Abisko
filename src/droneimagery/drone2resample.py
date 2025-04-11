#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drone Image Resolution Resampler

This script reads drone imagery GeoTIFFs, determines their native spatial 
resolution, and resamples them to a user-specified target resolution.
Useful for scale analysis, creating pyramids, or matching satellite resolution.

Author: Shunan Feng (shf@ign.ku.dk)
"""
#%%
import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio import warp 
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from rasterio.plot import show
# from rasterio.plot import reshape_as_image
import seaborn as sns
sns.set_theme(style="whitegrid", font_scale=1.5)
#%%
def resample_drone_image(input_path, output_path=None, target_resolution=None, 
                         resample_method='bilinear', format_options=None):
    """
    Resample a drone image to a specified target resolution.
    
    Parameters:
        input_path (str): Path to input drone GeoTIFF
        output_path (str): Path for resampled output (if None, auto-generated)
        target_resolution (float): Desired pixel size in dataset units (usually meters)
        resample_method (str): Resampling algorithm: 'nearest', 'bilinear', 'cubic', 
                               'cubic_spline', 'lanczos', 'average', 'mode', 'max', 'min',
                               'med', 'q1', 'q3', 'sum', 'rms'
        format_options (dict): Additional format-specific options for output

    Note: mode takes significantly longer to process than other methods
    
    Returns:
        str: Path to the resampled output file
    """
    # Validate input file
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Map resampling method string to rasterio enum
    resampling_methods = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic,
        'cubic_spline': Resampling.cubic_spline,
        'lanczos': Resampling.lanczos,
        'average': Resampling.average,
        'mode': Resampling.mode, # significantly slower, not recommended
        'max': Resampling.max,
        'min': Resampling.min,
        'med': Resampling.med,
        'q1': Resampling.q1,
        'q3': Resampling.q3,
        'sum': Resampling.sum,
        'gauss': Resampling.rms
    }
    
    if resample_method not in resampling_methods:
        raise ValueError(f"Invalid resampling method. Choose from: {', '.join(resampling_methods.keys())}")
    
    # Set default format options
    if format_options is None:
        format_options = {
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'interleave': 'band',
            'compress': 'lzw',
            'predictor': 3
        }
    
    with rasterio.open(input_path) as src:
        
        # Check units of the dataset
        if src.crs.is_projected:
            units = src.crs.linear_units
            print(f"Dataset is projected. Units: {units}")
        else:
            raise ValueError("Dataset is not projected. Cannot determine resolution. Assume meters.")
        
        # Get current resolution
        current_resolution_x = src.transform[0]  # Pixel width
        current_resolution_y = abs(src.transform[4])  # Pixel height (absolute value)
        
        # If target resolution is not provided, just print info and exit
        if target_resolution is None:
            print(f"Current resolution: {current_resolution_x} x {current_resolution_y} units")
            return None
        
        # Calculate scale factors
        scale_x = target_resolution / current_resolution_x
        scale_y = target_resolution / current_resolution_y
        
        # Only proceed if we're actually changing the resolution
        if scale_x == 1.0 and scale_y == 1.0:
            print("Target resolution is the same as current resolution. No resampling needed.")
            return input_path
        
        # Calculate new dimensions
        new_width = int(src.width / scale_x)
        new_height = int(src.height / scale_y)
        
        # Create output path if not provided
        if output_path is None:
            base_dir = os.path.dirname(input_path)
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            # Create a resampled subfolder
            resample_dir = os.path.join(base_dir, f"{base_name}_resampled_{resample_method}")
            os.makedirs(resample_dir, exist_ok=True)
            output_path = os.path.join(resample_dir, f"{base_name}_{target_resolution}m_{resample_method}.tif")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Create new transform
        new_transform = from_origin(
            src.transform.c, src.transform.f,  # Origin (upper left corner)
            target_resolution, target_resolution  # Pixel sizes
        )
        
        # Create output profile
        out_profile = src.profile.copy()
        out_profile.update({
            'height': new_height,
            'width': new_width,
            'transform': new_transform,
            **format_options
        })
        
        print(f"Resampling image from {current_resolution_x}m to {target_resolution}m resolution")
        print(f"Original size: {src.width} x {src.height} pixels")
        print(f"New size: {new_width} x {new_height} pixels")
        
        # Create the resampled dataset
        with rasterio.open(output_path, 'w', **out_profile) as dst:
            # Process each band
            for i in tqdm(range(1, src.count + 1), desc="Resampling bands"):
                # Read input band
                data = src.read(i)
                
                # Resample using the specified method
                resampled_data = np.empty((new_height, new_width), dtype=data.dtype)
                warp.reproject(
                    source=data,
                    destination=resampled_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=new_transform,
                    dst_crs=src.crs,
                    resampling=resampling_methods[resample_method]
                )
                
                # Write to destination
                dst.write(resampled_data, i)
        
        print(f"Resampled image saved to: {output_path}")
        return output_path

#%%
def visualize_resampled_image(resampled_path, bands=(3, 2, 1), figsize=(15, 15), resample_method=None):
    """
    Visualize only the resampled image with a scale bar.
    
    Parameters:
        resampled_path (str): Path to the resampled GeoTIFF
        bands (tuple): Bands to use for RGB visualization (1-based, not 0-based)
        figsize (tuple): Figure size
        resample_method (str): The resampling method used
    """
    # Get base output directory
    output_dir = os.path.dirname(resampled_path)
    
    # Make visualization directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    print("Creating resampled image visualization...")
    
    # Plot RGB of resampled image
    resampled_name = os.path.splitext(os.path.basename(resampled_path))[0]
    with rasterio.open(resampled_path) as src:
        resampled_res = src.transform[0]
        
        band_data = src.read(bands)
        # if any band data is nan, set it to nan
        # Create a mask where any band has NaN values
        nan_mask = np.isnan(band_data).any(axis=0)
        
        # Apply the mask to all bands
        for i in range(band_data.shape[0]):
            band_data[i][nan_mask] = np.nan
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Show the RGB image
        # show(rgb_norm, ax=ax, transform=src.transform)
        show(band_data, ax=ax, transform=src.transform)
        
        # Add scale bar
        scalebar = ScaleBar(
            1, # resampled_res, # rasterio already sets the scale, not needed when using rasterio
            'm',
            dimension='si-length',
            color='white',
            box_alpha=0.5,
            location='lower right'
            # font_properties={'size': 14}
        )
        ax.add_artist(scalebar)
        
        # Add title
        # ax.set_title(f"Resampled Image - {resampled_res:.2f}m Resolution")
        
        # Add metadata as text
        metadata_text = (
            f"Resampling Method: {resample_method if resample_method else 'Unknown'}\n"
            f"Resolution: {resampled_res:.2f}m\n"
            f"Image Size: {src.width} x {src.height} pixels\n"
            f"CRS: {src.crs.to_string()}"
        )

        ax.text(0.02, 0.02, metadata_text, transform=ax.transAxes, 
                verticalalignment='bottom', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        output_path_png = os.path.join(viz_dir, f"{resampled_name}.png")
        fig.savefig(output_path_png, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path_png}")
        output_path_pdf = os.path.join(viz_dir, f"{resampled_name}.pdf")
        fig.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path_pdf}")
        
        # plt.show()
        
        return fig, ax

#%%
# Update the main function to only visualize the resampled image
def main():
    """Main function to parse arguments and execute resampling across methods and resolutions"""
    input_path = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/3_Shunan/data/studentdebug/23_06_08_orthomosaic_georef_processed.tif"
    target_resolutions = [0.1, 0.5, 1, 2, 3, 5, 10, 15, 20, 30]  # list of resolutions in meters
    resample_methods = [
        # common resampling methods
        'nearest', 'bilinear', 'cubic', 
        'cubic_spline', 'lanczos', 
        'average', 'max', 'min',
        'med', 'q1', 'q3', 'sum', 'rms'
    ]  
    
    # Loop through each resampling method
    for resample_method in resample_methods:
        print(f"\n\nProcessing with resampling method: {resample_method}")
        
        # Loop through each target resolution
        for target_resolution in target_resolutions:
            print(f"\nProcessing target resolution: {target_resolution}m")
            
            # Perform resampling
            output_path = resample_drone_image(
                input_path=input_path, 
                target_resolution=target_resolution, 
                resample_method=resample_method)
            
            # Only visualize the resampled image if resampling was performed
            if output_path:
                visualize_resampled_image(output_path, resample_method=resample_method)
    # alternatively, process and visualize a single resampled image
    # output_path = resample_drone_image(input_path, target_resolution=0.02, resample_method='nearest')
    # visualize_resampled_image(output_path, resample_method='nearest')
if __name__ == "__main__":
    main()