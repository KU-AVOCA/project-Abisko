#!/usr/bin/env python3
# filepath: extract_polygon_stats.py
#%%
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import seaborn as sns
from itertools import combinations
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid", font_scale=1.5)
#%%
def extract_polygon_stats(raster_path, polygon_path, landcover_column="Classname"):
    """
    Extract mean and std of pixel values for each polygon, grouped by landcover class.
    
    Parameters:
    -----------
    raster_path : str
        Path to the drone imagery (GeoTIFF)
    polygon_path : str
        Path to polygon shapefile
    landcover_column : str
        Column name in the shapefile that contains landcover class information
        
    Returns:
    --------
    DataFrame with statistics for each landcover class
    """
    # Read the polygon file
    polygons = gpd.read_file(polygon_path)
    
    # Check if landcover column exists
    if landcover_column not in polygons.columns:
        raise ValueError(f"Landcover column '{landcover_column}' not found in shapefile")
    
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Get basic raster info
        bands = src.count
        
        # Initialize results storage
        results = []
        
        # Loop through each polygon
        for idx, poly in polygons.iterrows():
            # Get the geometry and class
            geom = poly.geometry
            lc_class = poly[landcover_column]
            
            # Skip if geometry is invalid or empty
            if not geom.is_valid or geom.is_empty:
                print(f"Warning: Invalid geometry in polygon {idx}, skipping")
                continue
            
            try:
                # Mask the raster with the polygon
                masked, mask_transform = mask(src, [mapping(geom)], crop=True, all_touched=True)
                
                # Process each band
                for band in range(bands):
                    # Get the masked data for this band
                    band_data = masked[band].flatten()
                    # Remove nodata values
                    band_data = band_data[band_data != src.nodata] if src.nodata is not None else band_data
                    
                    if len(band_data) > 0:
                        results.append({
                            'polygon_id': idx,
                            'landcover_class': lc_class,
                            'band': band + 1,  # 1-based band indexing
                            'mean': float(np.nanmean(band_data)),
                            'std': float(np.nanstd(band_data)),
                            'count': len(band_data)
                        })
            except Exception as e:
                print(f"Error processing polygon {idx}: {e}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Group by landcover class and band
    summary = results_df.groupby(['landcover_class', 'band']).agg({
        'mean': 'mean',
        'std': 'mean',  # Average of standard deviations
        'count': 'sum'
    }).reset_index()
    
    return results_df, summary

def main():
    """Main function to run the script"""
    # Set your input and output paths here
    raster_path = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/drone/orthomosaic4analysis/22_07_07_orthomosaic_georef_processed.tif"
    polygon_path = "Trainingdata_polygons_new_transfer.shp"
    class_column = "Classname"  # Change if your shapefile uses a different column name
    output_csv = "polygon_stats.csv"
    summary_csv = "class_summary.csv"
    
    # You can modify the above variables as needed
    
    class Args:
        pass
    args = Args()
    args.raster = raster_path
    args.polygons = polygon_path
    args.class_column = class_column
    args.output = output_csv
    args.summary_output = summary_csv
    
    print(f"Extracting statistics from {args.raster} using polygons from {args.polygons}...")
    
    # Run the extraction
    detail_results, summary_results = extract_polygon_stats(
        args.raster, args.polygons, args.class_column)
    
    # Save results
    detail_results.to_csv(args.output, index=False)
    summary_results.to_csv(args.summary_output, index=False)
    
    print(f"Detailed results saved to {args.output}")
    print(f"Summary results saved to {args.summary_output}")
    
    # Print summary
    print("\nSummary by landcover class:")
    for lc_class in summary_results['landcover_class'].unique():
        class_data = summary_results[summary_results['landcover_class'] == lc_class]
        print(f"\n{lc_class}:")
        for _, row in class_data.iterrows():
            print(f"  Band {int(row['band'])}: Mean={row['mean']:.2f}, StdDev={row['std']:.2f}")

if __name__ == "__main__":
    main()
# %% plot the spectral profile

df = pd.read_csv("polygon_stats.csv")
# convert band numbers to wavelengths
df["band"] = df["band"].map({
    1: 475,
    2: 560,
    3: 668,
    4: 717,
    5: 842
})
# band1: 475 nm
# band2: 560 nm
# band3: 668 nm
# band4: 717 nm
# band5: 842 nm
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=df, x="band", y="mean", hue="landcover_class", style="landcover_class",ax=ax, markers=True, dashes=False)
ax.legend(bbox_to_anchor=(1.01, 0.9), loc='upper left')
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Surface Reflectance")
# %% 
# %% Evaluate spectral similarity using Spectral Angle Mapper (SAM)

mean_spectral_profiles = df.groupby(['landcover_class', 'band'])['mean'].mean().reset_index()

# Pivot the table to get spectral profiles in a convenient format:
# Rows: landcover_class, Columns: bands (wavelengths), Values: mean reflectance
pivot_profiles = mean_spectral_profiles.pivot(index='landcover_class', columns='band', values='mean')

print("\nMean Spectral Profiles (Endmembers):")
print(pivot_profiles)

# Import necessary library for combinations

landcover_classes = pivot_profiles.index.tolist()
sam_results_list = []

if len(landcover_classes) < 2:
    print("\nNot enough landcover classes (need at least 2) to compare similarity.")
else:
    print("\nCalculating Spectral Angle Mapper (SAM) between landcover classes...")
    for lc1, lc2 in combinations(landcover_classes, 2):
        # Get spectral vectors for the two classes
        vec1 = pivot_profiles.loc[lc1].values.astype(float)
        vec2 = pivot_profiles.loc[lc2].values.astype(float)
        
        # Calculate SAM
        # SAM = arccos( (V1 . V2) / (||V1|| * ||V2||) )
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        # Check for zero norm vectors to avoid division by zero
        if norm_vec1 == 0 or norm_vec2 == 0:
            sam_angle_rad = np.pi / 2  # Assign max dissimilarity (90 degrees)
            print(f"Warning: Zero norm vector for '{lc1 if norm_vec1 == 0 else lc2}'. SAM set to 90 degrees for pair ('{lc1}', '{lc2}').")
        else:
            cosine_angle = dot_product / (norm_vec1 * norm_vec2)
            # Clip to avoid domain errors with arccos due to floating point inaccuracies
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            sam_angle_rad = np.arccos(cosine_angle)
        
        sam_angle_deg = np.degrees(sam_angle_rad)
        
        sam_results_list.append({
            'Class 1': lc1,
            'Class 2': lc2,
            'SAM (degrees)': sam_angle_deg
        })

    # Convert results to a DataFrame for better display
    sam_df = pd.DataFrame(sam_results_list)
    if not sam_df.empty:
        sam_df = sam_df.sort_values(by='SAM (degrees)').reset_index(drop=True)
        print("\nPairwise Spectral Similarity (SAM):")
        print("(Lower SAM values indicate higher similarity)")
        print(sam_df)

        # Create a SAM matrix for heatmap visualization
        sam_matrix = pd.DataFrame(index=landcover_classes, columns=landcover_classes, dtype=float)
        for _, row in sam_df.iterrows():
            sam_matrix.loc[row['Class 1'], row['Class 2']] = row['SAM (degrees)']
            sam_matrix.loc[row['Class 2'], row['Class 1']] = row['SAM (degrees)'] # Symmetric matrix
        
        # Fill diagonal with 0 (a class is perfectly similar to itself)
        np.fill_diagonal(sam_matrix.values, 0)

        print("\nSAM Matrix (degrees):")
        print(sam_matrix)

        # Plot heatmap of SAM values
        if not sam_matrix.empty:
            fig_sam, ax_sam = plt.subplots(figsize=(10, 8))
            sns.heatmap(sam_matrix, annot=True, fmt=".2f", cmap="viridis_r", ax=ax_sam, cbar_kws={'label': 'SAM (degrees)'})
            # 'viridis_r' colormap: lower values (more similar) are darker
            ax_sam.set_title("Spectral Angle Mapper (SAM) Between Landcover Classes")
            plt.tight_layout()
            plt.show()
    else:
        print("No SAM results to display.")
# %%
