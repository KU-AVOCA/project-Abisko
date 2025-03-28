#%%
import rasterio
from rasterio.plot import show
from rasterio.windows import Window  
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm 
import cmocean as cmo
from contextlib import nullcontext

#%%
def fill_gaps_by_interpolation(data, mask=None, method='linear'):
    """
    Fill gaps in data array using interpolation
    
    Parameters:
        data (ndarray): Input data array with NaN values
        mask (ndarray): Boolean mask of pixels to interpolate (True = fill these)
        method (str): Interpolation method: 'linear', 'nearest', 'cubic'
        
    Returns:
        ndarray: Array with gaps filled
    """
    from scipy import ndimage
    from scipy.interpolate import griddata
    
    # If no mask provided, use NaN locations
    if mask is None:
        mask = np.isnan(data)
    
    # If nothing to fill, return original data
    if not np.any(mask):
        return data
    
    # Make a copy to avoid modifying the original
    filled_data = data.copy()
    
    # Get valid data points and their coordinates
    valid_mask = ~np.isnan(filled_data)
    
    # If not enough valid points, use nearest neighbor via distance transform
    if np.sum(valid_mask) < 4:  # Need at least a few points for interpolation
        # Simple nearest neighbor filling using distance transform
        if np.any(valid_mask):
            # Use distance transform to find nearest non-NaN value
            indices = ndimage.distance_transform_edt(
                ~valid_mask, return_distances=False, return_indices=True
            )
            filled_data[mask] = filled_data[valid_mask][indices[0][mask], indices[1][mask]]
        return filled_data
    
    # Get coordinates of valid and invalid points
    y, x = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    valid_points = np.column_stack((y[valid_mask], x[valid_mask]))
    valid_values = filled_data[valid_mask]
    
    # Points to interpolate
    interp_points = np.column_stack((y[mask], x[mask]))
    
    try:
        # Use griddata for interpolation
        if method == 'nearest':
            interp_method = 'nearest'
        elif method == 'cubic':
            interp_method = 'cubic'
        else:  # Default to linear
            interp_method = 'linear'
            
        # Interpolate
        filled_values = griddata(
            valid_points, valid_values, interp_points, 
            method=interp_method, fill_value=np.nan
        )
        
        # Fill any remaining NaNs with nearest neighbor
        if np.any(np.isnan(filled_values)):
            nn_values = griddata(
                valid_points, valid_values, interp_points[np.isnan(filled_values)],
                method='nearest'
            )
            filled_values[np.isnan(filled_values)] = nn_values
        
        # Update the array
        filled_data[mask] = filled_values
        
    except Exception as e:
        print(f"  Interpolation error: {e}. Using nearest neighbor method.")
        # Fallback to simple nearest neighbor via distance transform
        indices = ndimage.distance_transform_edt(
            ~valid_mask, return_distances=False, return_indices=True
        )
        filled_data[mask] = filled_data[valid_mask][indices[0][mask], indices[1][mask]]
    
    return filled_data

def process_altum_orthomosaic(input_path, output_dir=None, create_mask=True, fill_gaps=False, interp_method='linear'):
    """
    Process ALTUM orthomosaic using block processing to handle large files
    
    Parameters:
        input_path (str): Path to input ALTUM orthomosaic
        output_dir (str): Output directory for processed files (default: same as input)
        create_mask (bool): Whether to generate a mask file
        fill_gaps (bool): Whether to fill data gaps using interpolation
        interp_method (str): Interpolation method ('linear', 'nearest', 'cubic')
        
    Returns:
        tuple: (ms_path, thermal_path, mask_path) - Paths to output files
    """
    # Setup output paths
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = os.path.splitext(os.path.basename(input_path))[0]
    ms_output_path = os.path.join(output_dir, f"{base_filename}_processed.tif")
    thermal_output_path = os.path.join(output_dir, f"{base_filename}_thermal_celsius.tif")
    mask_output_path = os.path.join(output_dir, f"{base_filename}_masks.tif") if create_mask else None
    
    # Check existing files
    for file_path in [ms_output_path, thermal_output_path, mask_output_path]:
        if file_path and os.path.exists(file_path):
            print(f"Warning: File {file_path} already exists and will be overwritten.")
    
    print(f"Processing {input_path} using block processing...")
    
    with rasterio.open(input_path) as src:
        # Read metadata
        profile = src.profile.copy()
        band_count = src.count
        width = src.width
        height = src.height
        dtype = profile['dtype']
        scaling_factor = 10000
        
        print(f"Image has {band_count} bands, dimensions: {width}x{height}, dtype: {dtype}")
        
        if band_count < 6:
            raise ValueError("Expected at least 6 bands for ALTUM data (5 MS + thermal)")
        
        # Setup output profiles
        ms_profile = profile.copy()
        ms_profile.update({
            'count': band_count - 1,  # Exclude thermal band
            'dtype': 'float32',
            'nodata': np.nan,
            'BIGTIFF': 'YES',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'interleave': 'band',    
            'compress': 'lzw',  
            'Predictor': 3
        })
        
        thermal_profile = profile.copy()
        thermal_profile.update({
            'count': 1,
            'dtype': 'float32',
            'nodata': np.nan,
            'BIGTIFF': 'YES',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'interleave': 'band',    
            'compress': 'lzw', 
            'Predictor': 3     
        })
        
        # Setup mask profile if needed
        if create_mask:
            mask_profile = profile.copy()
            mask_profile.update({
                'count': band_count,  # One mask per band + background mask
                'dtype': 'uint8',    # 0=valid, 1=invalid
                'nodata': None,
                'BIGTIFF': 'YES',
                'tiled': True,
                'blockxsize': 256,
                'blockysize': 256,
                'interleave': 'band',    
                'compress': 'lzw'
            })
        
        # Define chunk size for efficient processing
        chunk_size = 1024  # Adjust based on memory constraints
        
        # Create output files
        with rasterio.open(ms_output_path, 'w', **ms_profile) as ms_dst, \
             rasterio.open(thermal_output_path, 'w', **thermal_profile) as thermal_dst, \
             (rasterio.open(mask_output_path, 'w', **mask_profile) if create_mask else nullcontext()) as mask_dst:
            
            # Store background masks in memory instead of reading from file
            if create_mask:
                print("Creating background mask...")
                # Initialize in-memory background mask storage
                background_masks = {}
                
                for y in tqdm(range(0, height, chunk_size), desc="Processing background mask"):
                    y_size = min(chunk_size, height - y)
                    for x in range(0, width, chunk_size):
                        x_size = min(chunk_size, width - x)
                        window = Window(col_off=x, row_off=y, width=x_size, height=y_size)
                        
                        # Check all bands for zeros to identify true background
                        is_background = np.zeros((y_size, x_size), dtype=bool)
                        
                        for i in range(1, band_count + 1):
                            block = src.read(i, window=window)
                            is_background = is_background | (block == 0)
                        
                        # Write background mask (band 1)
                        mask_dst.write(is_background.astype(np.uint8), 1, window=window)
                        
                        # Store mask in memory for later use (using window position as key)
                        background_masks[(y, x)] = is_background
            
            # Process multispectral bands
            for i in tqdm(range(1, band_count), desc="Processing multispectral bands"):
                print(f"Processing multispectral band {i}...")
                
                # Process image in chunks
                for y in range(0, height, chunk_size):
                    y_size = min(chunk_size, height - y)
                    for x in range(0, width, chunk_size):
                        x_size = min(chunk_size, width - x)
                        
                        # Define window
                        window = Window(col_off=x, row_off=y, width=x_size, height=y_size)
                        
                        # Read block data
                        block_data = src.read(i, window=window).astype("float32")
                        
                        # Create mask for this block
                        if create_mask:
                            # Use background mask from memory instead of reading from file
                            background_mask = background_masks[(y, x)]
                            
                            # Identify invalid pixels (0s or values ≥ 1 after scaling)
                            invalid_mask = (block_data <= 0) | (block_data >= scaling_factor)
                            
                            # Write band-specific mask (background is already in band 1)
                            mask_dst.write(invalid_mask.astype(np.uint8), i + 1, window=window)
                        else:
                            invalid_mask = (block_data <= 0) | (block_data >= scaling_factor)
                        
                        # Process block
                        block_data = block_data / scaling_factor
                        
                        # Mask invalid pixels
                        block_data[invalid_mask] = np.nan
                        
                        # Fill gaps if requested
                        if fill_gaps and np.any(np.isnan(block_data)):
                            if create_mask:
                                # Only interpolate non-background NaN values
                                interpolate_mask = np.isnan(block_data) & ~background_mask
                                if np.any(interpolate_mask):
                                    print(f"  Band {i}, Block ({y},{x}): Filling {np.sum(interpolate_mask)} data gaps")
                                    block_data = fill_gaps_by_interpolation(block_data, interpolate_mask, method=interp_method)
                            else:
                                # Interpolate all NaN values
                                interpolate_mask = np.isnan(block_data)
                                if np.any(interpolate_mask):
                                    print(f"  Band {i}, Block ({y},{x}): Filling {np.sum(interpolate_mask)} data gaps")
                                    block_data = fill_gaps_by_interpolation(block_data, interpolate_mask, method=interp_method)
                        
                        # Write block to output
                        ms_dst.write(block_data, i, window=window)
            
            print("Finished processing multispectral bands")
            
            # Process thermal band
            print("Processing thermal band...")
            
            # Process thermal band in chunks
            for y in range(0, height, chunk_size):
                y_size = min(chunk_size, height - y)
                for x in range(0, width, chunk_size):
                    x_size = min(chunk_size, width - x)
                    
                    # Define window
                    window = Window(col_off=x, row_off=y, width=x_size, height=y_size)
                    
                    # Read block data
                    thermal_block = src.read(6, window=window).astype("float32")
                    
                    # Create mask for this block
                    if create_mask:
                        # Use background mask from memory instead of reading from file
                        background_mask = background_masks[(y, x)]
                        
                        # Identify invalid pixels
                        invalid_mask = (thermal_block == 65535) | (thermal_block <= 0)
                        
                        # Write thermal mask
                        mask_dst.write(invalid_mask.astype(np.uint8), band_count, window=window)
                    else:
                        invalid_mask = (thermal_block == 65535) | (thermal_block <= 0)
                    
                    # Process block
                    thermal_block[invalid_mask] = np.nan
                    thermal_block = thermal_block / 100 - 273.15
                    
                    # Fill gaps if requested
                    if fill_gaps and np.any(np.isnan(thermal_block)):
                        if create_mask:
                            # Only interpolate non-background NaN values
                            interpolate_mask = np.isnan(thermal_block) & ~background_mask
                            if np.any(interpolate_mask):
                                print(f"  Thermal band, Block ({y},{x}): Filling {np.sum(interpolate_mask)} data gaps")
                                thermal_block = fill_gaps_by_interpolation(thermal_block, interpolate_mask, method=interp_method)
                        else:
                            # Interpolate all NaN values
                            interpolate_mask = np.isnan(thermal_block)
                            if np.any(interpolate_mask):
                                print(f"  Thermal band, Block ({y},{x}): Filling {np.sum(interpolate_mask)} data gaps")
                                thermal_block = fill_gaps_by_interpolation(thermal_block, interpolate_mask, method=interp_method)
                    
                    # Write block to output
                    thermal_dst.write(thermal_block, 1, window=window)
            
            print("Finished processing thermal band")
        
        print(f"Processed files saved to:")
        print(f"  - Multispectral: {ms_output_path}")
        print(f"  - Thermal: {thermal_output_path}")
        if create_mask:
            print(f"  - Mask: {mask_output_path}")
            
            # Add description to mask bands
            with rasterio.open(mask_output_path, 'r+') as mask_src:
                mask_src.set_band_description(1, 'Background mask')
                for i in range(1, band_count):
                    mask_src.set_band_description(i+1, f'Band {i} invalid mask')
                mask_src.set_band_description(band_count, 'Thermal invalid mask')
        
        # Return paths to the created files
        return ms_output_path, thermal_output_path, mask_output_path if create_mask else None

def visualize_results(ms_path, thermal_path, downsample_factor):
    """
    Create visualization of the processed data with downsampling for large files
    
    Args:
        ms_path (str): Path to processed multispectral image
        thermal_path (str): Path to processed thermal image
        downsample_factor (int): Factor by which to downsample for visualization
    """
    print(f"Creating visualizations with downsample factor {downsample_factor}...")
    output_dir = os.path.dirname(ms_path)
    base_name = os.path.splitext(os.path.basename(ms_path))[0]
    
    # Band names for labels
    band_names = ['Blue', 'Green', 'Red', 'Red Edge', 'NIR']
    
    # 1. Save individual band visualizations
    with rasterio.open(ms_path) as src:
        # Calculate downsampled dimensions
        out_height = int(src.height * downsample_factor)
        out_width = int(src.width * downsample_factor)
        
        for i in range(1, 6):
            print(f"Creating {band_names[i-1]} band visualization...")
            band_fig, band_ax = plt.subplots(figsize=(10, 10))
            
            # Read single band with downsampling
            band_data = src.read(
                i, out_shape=(1, out_height, out_width), resampling=rasterio.enums.Resampling.bilinear
            )
            
            # Show band with appropriate colormap

            if i == 1: # Blue band  
                cmap = 'Blues'
            elif i == 2: # Green band
                cmap = 'Greens'
            elif i == 3: # Red band
                cmap = 'Reds'
            elif i == 4: # Red Edge band
                cmap = 'Oranges'
            else: # NIR band
                cmap = cmo.cm.haline
                
            show(band_data, ax=band_ax, cmap=cmap, title=f"{band_names[i-1]} Band")
            band_ax.set_title(f"{band_names[i-1]} Band")
            band_ax.axis('off')
            
            # Add colorbar
            im = band_ax.imshow(band_data, cmap=cmap)
            plt.colorbar(im, ax=band_ax, shrink=0.7, label=f'Reflectance')
            
            # Save band visualization
            band_path = os.path.join(output_dir, f"{base_name}_{band_names[i-1].lower()}.png")
            plt.savefig(band_path, dpi=300, bbox_inches='tight')
            plt.close(band_fig)
            print(f"{band_names[i-1]} band visualization saved to: {band_path}")
            
            # Free memory
            del band_data
    
    # 2. Save RGB visualization separately
    print("Creating true color (RGB) visualization...")
    with rasterio.open(ms_path) as src:
        # Create a new figure for RGB only
        rgb_fig, rgb_ax = plt.subplots(figsize=(10, 10))
        
        # For true color (RGB), we want to use Red (band 3), Green (band 2), Blue (band 1)
        print("Reading bands for RGB visualization...")
        rgb_bands = src.read([3, 2, 1], out_shape=(3, out_height, out_width), resampling=rasterio.enums.Resampling.bilinear)
        
        # Show true color image
        show(rgb_bands, ax=rgb_ax, title="True Color (RGB)")
        rgb_ax.set_title('True Color (RGB)')
        rgb_ax.axis('off')
        
        # Save RGB visualization
        rgb_path = os.path.join(output_dir, f"{base_name}_rgb.png")
        plt.savefig(rgb_path, dpi=300, bbox_inches='tight')
        plt.close(rgb_fig)
        print(f"RGB visualization saved to: {rgb_path}")
        
        # Free memory
        del rgb_bands
    
    # 3. Save CIR (Color Infrared) visualization
    print("Creating color infrared (CIR) visualization...")
    with rasterio.open(ms_path) as src:
        # Create a new figure for CIR only
        cir_fig, cir_ax = plt.subplots(figsize=(10, 10))
        
        # For color infrared, use NIR (band 5), Red (band 3), Green (band 2)
        print("Reading bands for CIR visualization...")
        cir_bands = src.read([5, 3, 2], out_shape=(3, out_height, out_width), resampling=rasterio.enums.Resampling.bilinear)
        
        # Show CIR image
        show(cir_bands, ax=cir_ax, title="Color Infrared (CIR)")
        cir_ax.set_title('Color Infrared (CIR)')
        cir_ax.axis('off')
        
        # Save CIR visualization
        cir_path = os.path.join(output_dir, f"{base_name}_cir.png")
        plt.savefig(cir_path, dpi=300, bbox_inches='tight')
        plt.close(cir_fig)
        print(f"CIR visualization saved to: {cir_path}")
        
        # Free memory
        del cir_bands
    
    # 4. Save thermal visualization separately
    print("Creating thermal visualization...")
    with rasterio.open(thermal_path) as src:
        # Create a new figure for thermal only
        thermal_fig, thermal_ax = plt.subplots(figsize=(10, 10))
        
        # Show thermal band with colormap
        thermal_data = src.read(1, out_shape=(1, out_height, out_width), resampling=rasterio.enums.Resampling.bilinear)
        show(thermal_data, ax=thermal_ax, cmap=cmo.cm.thermal, title="Thermal (°C)")
        thermal_ax.set_title('Thermal (°C)')
        thermal_ax.axis('off')
        
        # Add colorbar
        im = thermal_ax.imshow(thermal_data, cmap=cmo.cm.thermal)
        plt.colorbar(im, ax=thermal_ax, shrink=0.7, label='Temperature (°C)')
        
        # Save thermal visualization
        thermal_img_path = os.path.join(output_dir, f"{base_name}_thermal.png")
        plt.savefig(thermal_img_path, dpi=300, bbox_inches='tight')
        plt.close(thermal_fig)
        print(f"Thermal visualization saved to: {thermal_img_path}")
        
        # Free memory
        del thermal_data
    
    print("All visualizations complete.")

#%% Example usage
if __name__ == "__main__":
    input_image = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/3_Shunan/data/studentdebug/23_06_08_orthomosaic_georef.tif"  # Replace with your file path
    output_directory = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/3_Shunan/data/studentdebug/"  # Optional, replace with your preferred output directory
    
    # Process the image with masking and gap filling
    ms_output, thermal_output, mask_output = process_altum_orthomosaic(
        input_image, 
        output_directory,
        create_mask=True,        # Generate mask file
        fill_gaps=True,          # Fill data gaps
        interp_method='linear'   # Use linear interpolation
    )
    ms_output, thermal_output, mask_output = process_altum_orthomosaic(input_image, output_directory)
    # ms_output = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/4_Student projects/1_Monika_Kathrine/2_Results/1_DroneImagery/postprocessed/20230608_orthomosaic32022_processed.tif"
    # thermal_output = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/4_Student projects/1_Monika_Kathrine/2_Results/1_DroneImagery/postprocessed/20230608_orthomosaic32022_thermal_celsius.tif"
    
    # Visualize the results
    # visualize_results(ms_output, thermal_output, downsample_factor=0.1)
# %%

