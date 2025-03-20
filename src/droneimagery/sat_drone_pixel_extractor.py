"""
Script to extract drone pixel values based on satellite pixel footprints.
For each pixel in a satellite image, this script:
1. Determines the geographic footprint
2. Finds all drone pixels within that footprint
3. Calculates statistics for those drone pixels (mean, median, etc.)
4. Saves the satellite pixel value and drone pixel statistics to CSV files (one per band)

Useful for validating satellite measurements with drone data or analyzing sub-pixel heterogeneity.

Shunan Feng (shf@ign.ku.dk)
"""
#%%
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import rowcol
import geopandas as gpd
from shapely.geometry import box, Point
import matplotlib.pyplot as plt
from tqdm import tqdm

#%%
def extract_pixels(drone_tif, satellite_tif, output_dir, sample_fraction=1.0, max_pixels=10000):
    """
    Extract drone pixel values within each satellite pixel footprint
    and save results to CSV files.
    
    Parameters:
        drone_tif (str): Path to drone GeoTIFF file
        satellite_tif (str): Path to satellite GeoTIFF file
        output_dir (str): Directory to save output CSV files
        sample_fraction (float): Fraction of satellite pixels to process (0-1)
        max_pixels (int): Maximum number of satellite pixels to process
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get satellite image basename for output files
    sat_basename = os.path.splitext(os.path.basename(satellite_tif))[0]
    
    # Open the drone and satellite images
    with rasterio.open(drone_tif) as drone_src, rasterio.open(satellite_tif) as sat_src:
        # Check if they're in the same CRS
        if drone_src.crs != sat_src.crs:
            raise ValueError(f"CRS mismatch: Drone ({drone_src.crs}) vs Satellite ({sat_src.crs})")
        
        # Get number of bands for each image
        drone_bands = drone_src.count
        sat_bands = sat_src.count
        
        # Get band names if available
        drone_bandnames = [drone_src.descriptions[i-1] or f"Band_{i}" for i in range(1, drone_bands+1)]
        sat_bandnames = [sat_src.descriptions[i-1] or f"Band_{i}" for i in range(1, sat_bands+1)]
        
        print(f"Drone image: {drone_bands} bands")
        print(f"Satellite image: {sat_bands} bands")
        
        # Get satellite pixel dimensions in same units as the CRS (usually meters)
        sat_pixel_width = sat_src.transform[0]
        sat_pixel_height = -sat_src.transform[4]  # Negative because origin is top-left
        
        # Calculate total number of satellite pixels
        total_pixels = sat_src.width * sat_src.height
        
        # Determine how many pixels to process based on sample fraction and max_pixels
        num_pixels_to_process = min(int(total_pixels * sample_fraction), max_pixels)
        
        print(f"Total satellite pixels: {total_pixels}")
        print(f"Processing {num_pixels_to_process} pixels ({num_pixels_to_process/total_pixels*100:.1f}%)")
        
        # Create random sample of pixel indices if not processing all
        if num_pixels_to_process < total_pixels:
            # Generate random row, col pairs
            rows = np.random.randint(0, sat_src.height, size=num_pixels_to_process)
            cols = np.random.randint(0, sat_src.width, size=num_pixels_to_process)
            pixel_indices = list(zip(rows, cols))
        else:
            # Process all pixels
            pixel_indices = [(row, col) for row in range(sat_src.height) for col in range(sat_src.width)]
        
        # Process each band
        for sat_band_idx in range(1, sat_bands+1):
            print(f"Processing satellite band {sat_band_idx} ({sat_bandnames[sat_band_idx-1]})...")
            
            # Find matching drone band (if possible)
            # This is a simple implementation - you might need a more sophisticated mapping
            if sat_band_idx <= drone_bands:
                drone_band_idx = sat_band_idx
            else:
                print(f"  Warning: Satellite band {sat_band_idx} has no direct drone equivalent. Using drone band 1.")
                drone_band_idx = 1
            
            # Read full satellite band
            sat_band = sat_src.read(sat_band_idx)
            
            # Create list to store results
            results = []
            
            # Process each sampled satellite pixel
            for row, col in tqdm(pixel_indices, desc=f"Processing Band {sat_band_idx}"):
                # Skip if satellite pixel is nodata
                sat_value = sat_band[row, col]
                if sat_value == sat_src.nodata:
                    continue
                
                # Get satellite pixel bounds
                # Convert pixel indices to world coordinates
                x, y = sat_src.xy(row, col)
                
                # Calculate pixel bounds
                xmin = x - sat_pixel_width/2
                xmax = x + sat_pixel_width/2
                ymin = y - sat_pixel_height/2
                ymax = y + sat_pixel_height/2
                
                # Create a window for the drone image
                drone_window = from_bounds(xmin, ymin, xmax, ymax, drone_src.transform)
                
                # Read the drone pixels in this window
                drone_data = drone_src.read(drone_band_idx, window=drone_window)
                
                # Skip if no valid drone data
                if drone_data.size == 0 or (drone_src.nodata is not None and np.all(drone_data == drone_src.nodata)):
                    continue
                
                # Calculate statistics on drone pixels
                drone_pixel_count = drone_data.size
                drone_valid_pixels = drone_data[drone_data != drone_src.nodata] if drone_src.nodata is not None else drone_data
                
                # Skip if no valid pixels
                if drone_valid_pixels.size == 0:
                    continue
                    
                # Calculate statistics
                drone_mean = np.mean(drone_valid_pixels)
                drone_median = np.median(drone_valid_pixels)
                drone_min = np.min(drone_valid_pixels)
                drone_max = np.max(drone_valid_pixels)
                drone_std = np.std(drone_valid_pixels)
                
                # Store results
                results.append({
                    'sat_row': row,
                    'sat_col': col,
                    'sat_x': x,
                    'sat_y': y,
                    'sat_value': sat_value,
                    'drone_pixel_count': drone_pixel_count,
                    'drone_valid_pixels': drone_valid_pixels.size,
                    'drone_mean': drone_mean,
                    'drone_median': drone_median,
                    'drone_min': drone_min,
                    'drone_max': drone_max,
                    'drone_std': drone_std
                })
            
            # Create DataFrame and save to CSV
            if results:
                df = pd.DataFrame(results)
                
                # Get band names for the filename
                sat_band_name = sat_bandnames[sat_band_idx-1].replace(' ', '_').lower()
                drone_band_name = drone_bandnames[drone_band_idx-1].replace(' ', '_').lower()
                
                csv_filename = f"{sat_basename}_{sat_band_name}_vs_drone_{drone_band_name}.csv"
                csv_path = os.path.join(output_dir, csv_filename)
                
                df.to_csv(csv_path, index=False)
                print(f"Saved {len(df)} pixel comparisons to {csv_path}")
            else:
                print(f"No valid pixel comparisons found for band {sat_band_idx}")

        print("Processing complete!")

#%%
def visualize_comparison(csv_path, output_dir=None):
    """
    Create visualizations of the satellite vs. drone pixel comparisons.
    
    Parameters:
        csv_path (str): Path to CSV file containing pixel comparisons
        output_dir (str): Directory to save output visualizations (default: same as CSV)
    """
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(csv_path))[0]
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Create scatter plot of satellite values vs drone mean values
    plt.figure(figsize=(10, 8))
    plt.scatter(df['sat_value'], df['drone_mean'], alpha=0.5)
    plt.xlabel('Satellite Pixel Value')
    plt.ylabel('Drone Mean Pixel Value')
    plt.title('Satellite vs Drone Pixel Values')
    
    # Add a 1:1 line
    min_val = min(df['sat_value'].min(), df['drone_mean'].min())
    max_val = max(df['sat_value'].max(), df['drone_mean'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
    
    # Add regression line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['sat_value'], df['drone_mean'])
    plt.plot(df['sat_value'], slope * df['sat_value'] + intercept, 'g-', 
             label=f'Regression Line (r²={r_value**2:.3f})')
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f"{base_filename}_scatter.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved scatter plot to {plot_path}")
    
    # Create histogram of drone pixel standard deviations (sub-pixel heterogeneity)
    plt.figure(figsize=(10, 6))
    plt.hist(df['drone_std'], bins=50)
    plt.xlabel('Standard Deviation of Drone Pixels')
    plt.ylabel('Frequency')
    plt.title('Sub-pixel Heterogeneity')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    hist_path = os.path.join(output_dir, f"{base_filename}_heterogeneity.png")
    plt.savefig(hist_path, dpi=300)
    print(f"Saved heterogeneity histogram to {hist_path}")
    
    # Create some summary statistics
    print("\nSummary Statistics:")
    print(f"Number of pixels compared: {len(df)}")
    print(f"Correlation coefficient (r): {r_value:.4f}, r²: {r_value**2:.4f}")
    print(f"Mean drone pixels per satellite pixel: {df['drone_valid_pixels'].mean():.1f}")
    print(f"Mean satellite value: {df['sat_value'].mean():.4f}")
    print(f"Mean drone value: {df['drone_mean'].mean():.4f}")
    print(f"Mean sub-pixel heterogeneity (std): {df['drone_std'].mean():.4f}")
    
    # Return the DataFrame for further analysis
    return df

#%%
# Example usage
if __name__ == "__main__":
    # Set paths
    drone_tif = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/4_Student projects/1_Monika_Kathrine/2_Results/1_DroneImagery/postprocessed/20230608_orthomosaic32022_processed.tif"
    satellite_tif = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/4_Student projects/1_Monika_Kathrine/2_Results/1_DroneImagery/satellite/Sentinel2_20230608_0.tif"
    output_dir = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/4_Student projects/1_Monika_Kathrine/2_Results/1_DroneImagery/pixel_comparison"
    
    # Extract pixels and save to CSV
    extract_pixels(
        drone_tif=drone_tif,
        satellite_tif=satellite_tif,
        output_dir=output_dir,
        sample_fraction=0.1,  # Process 10% of pixels to keep runtime reasonable
        max_pixels=5000  # Cap at 5000 pixels
    )
    
    # Visualize results from one of the generated CSVs
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
    if csv_files:
        visualize_comparison(os.path.join(output_dir, csv_files[0]))

# %%