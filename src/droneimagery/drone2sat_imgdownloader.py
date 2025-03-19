#%%
import os
from glob import glob
import geopandas as gpd
import rasterio
from shapely.geometry import box
import geemap
import ee
import matplotlib.pyplot as plt
import numpy as np

"""
Script to extract extent from a GeoTIFF (drone orthomosaic), convert to shapefile, 
and download matching Landsat and Sentinel-2 imagery from Google Earth Engine for the ROI.
Cloud masking is applied using Cloud Score Plus (for Sentinel-2) and QA bands (for Landsat).
Bands are renamed to standardized names (blue, green, red, etc.) for easier processing.
"""


def tiff_to_shapefile(tiff_path, output_shp):
    """
    Extract the extent of a GeoTIFF file and create a shapefile boundary.
    
    Parameters:
        tiff_path (str): Path to the GeoTIFF file
        output_shp (str): Path to output shapefile (if None, will be created from input path)
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the extent polygon
    """
    # If output path is not specified, create one based on input
    if output_shp is None:
        output_shp = os.path.splitext(tiff_path)[0] + '_extent.shp'
    
    # Read the raster metadata
    with rasterio.open(tiff_path) as src:
        # Get the bounding box
        left, bottom, right, top = src.bounds
        # Get the CRS
        crs = src.crs
    
    # Create a polygon from the bounds
    polygon = box(left, bottom, right, top)
    
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs=crs)
    
    # Save to shapefile
    gdf.to_file(output_shp)
    
    print(f"Shapefile created: {output_shp}")
    return gdf


def maskS2sr(image):
    """
    Apply cloud masking to Sentinel-2 imagery using Cloud Score Plus.
    
    Parameters:
        image (ee.Image): Sentinel-2 image to mask
        
    Returns:
        ee.Image: Cloud-masked image with values scaled to 0-1 range
    """
    QA_BAND = 'cs'
    CLEAR_THRESHOLD = 0.65  # Cloud Score Plus threshold (higher = more strict)

    # Create mask for non-saturated pixels and non-cloud pixels
    not_saturated = image.select('SCL').neq(0)
    not_cloud = image.select(QA_BAND).gte(CLEAR_THRESHOLD)

    # Apply masks and scale values
    return image.updateMask(not_cloud).updateMask(not_saturated).divide(10000)

def renameS2(img):
    """
    Rename Sentinel-2 bands to standardized names for easier processing.
    
    Parameters:
        img (ee.Image): Sentinel-2 image with original band names
        
    Returns:
        ee.Image: Image with renamed bands
    """
    return img.select(
        ['B1',       'B2',   'B3',    'B4',  'B5',         'B6',         'B7',         'B8',  'B8A',        'B9',          'B11',   'B12',   'QA60', 'SCL'],
        ['aerosols', 'blue', 'green', 'red', 'red_edge_1', 'red_edge_2', 'red_edge_3', 'nir', 'red_edge_4', 'water_vapor', 'swir1', 'swir2', 'QA60', 'SCL']
    )

# Available Sentinel-2 bands:
# B1: Aerosols (60m)
# B2: Blue (10m)
# B3: Green (10m)
# B4: Red (10m)
# B5: Red Edge 1 (20m)
# B6: Red Edge 2 (20m)
# B7: Red Edge 3 (20m)
# B8: NIR (10m)
# B8A: Narrow NIR (20m)
# B9: Water vapor (60m)
# B11: SWIR 1 (20m)
# B12: SWIR 2 (20m)
# QA60: Cloud mask
# SCL: Scene classification

def prepS2(img):
    """
    Prepare Sentinel-2 imagery by applying cloud mask and renaming bands.
    
    Parameters:
        img (ee.Image): Raw Sentinel-2 image
        
    Returns:
        ee.Image: Processed image with cloud masking and standardized band names
    """
    orig = img
    img = maskS2sr(img)  # Apply cloud masking
    img = renameS2(img)  # Rename bands to standardized names

    # Preserve original metadata and add satellite identifier
    return ee.Image(img.copyProperties(orig, orig.propertyNames()).set('SATELLITE', 'SENTINEL_2'))

def download_sentinel2(shp_path, start_date, end_date, crs,output_dir):
    """
    Download Sentinel-2 imagery for the given ROI with Cloud Score Plus cloud masking.
    
    Parameters:
        shp_path (str): Path to shapefile defining the ROI
        start_date (ee.Date): Start date for image search
        end_date (ee.Date): End date for image search
        output_dir (str): Directory to save the downloaded images
        crs (str): Coordinate Reference System to use for the exported images
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert shapefile to Earth Engine geometry
    roi_ee = geemap.shp_to_ee(shp_path)
    roi = roi_ee.geometry()

    # Create filter for image collection
    s2colFilter = ee.Filter.And(
        ee.Filter.bounds(roi_ee), 
        ee.Filter.date(start_date, end_date)
    )
    
    # Get Sentinel-2 collection with Cloud Score Plus data
    # Cloud Score Plus provides cloud probability scores
    s2Col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .linkCollection(ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'), ['cs']) \
            .filter(s2colFilter) \
            .map(prepS2)   

    # Get the image count
    count = s2Col.size().getInfo()
    print(f"Found {count} Sentinel-2 images")
    
    if count == 0:
        print("No Sentinel-2 images found!")
        return
    else:
        print(f"Processing {count} Sentinel-2 images")
        
        # Download images
        images = s2Col.toList(count)

        for i in range(count):
            image = ee.Image(images.get(i))

            # Clip to ROI
            image_clipped = image.clip(roi_ee)
            
            # Create RGB visualization parameters
            vis_params = {
                'bands': ['red', 'green', 'blue'],
                'min': 0,
                'max': 0.3,
                'gamma': 1.4
            }
            
            # Add RGB composite to the map for visualization
            Map.addLayer(image_clipped, vis_params, f"Sentinel-2 RGB {i}")
            
            # Format date for filename
            date_acquired = ee.Date(image.get('system:time_start')).format('YYYYMMdd').getInfo()
            filename = f"Sentinel2_{date_acquired}_{i}.tif"
            output_path = os.path.join(output_dir, filename)
            
            # Export the image
            print(f"Downloading Sentinel-2 image: {filename}")
            geemap.ee_export_image(
                image_clipped, 
                filename=output_path,
                scale=10,  # Sentinel-2 RGB resolution (10m)
                region=roi,
                file_per_band=True,  # Save each band as a separate file
                crs=crs
            )


def maskL8sr(image):
    """
    Apply cloud masking to Landsat 8/9 imagery using the QA_PIXEL band.
    
    Parameters:
        image (ee.Image): Landsat image to mask
        
    Returns:
        ee.Image: Cloud-masked image with properly scaled band values
    """
    # Get the QA band for cloud masking
    qaMask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
    saturationMask = image.select('QA_RADSAT').eq(0)  # Mask saturated pixels
    
    # Apply scaling factors to convert DN to reflectance
    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2).updateMask(qaMask).updateMask(saturationMask)
    thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0).updateMask(qaMask).updateMask(saturationMask)

    # Return the masked and scaled image
    return image.addBands(opticalBands, None, True) \
            .addBands(thermalBands, None, True)

def renameOli(img):
    """
    Rename Landsat 8/9 bands to standardized names for easier processing.
    
    Parameters:
        img (ee.Image): Landsat image with original band names
        
    Returns:
        ee.Image: Image with renamed bands
    """
    return img.select(
        ['SR_B1',      'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10',   'QA_PIXEL', 'QA_RADSAT'],
        ['ultra_blue', 'blue',  'green', 'red',   'nir',   'swir1', 'swir2', 'thermal1', 'QA_PIXEL', 'QA_RADSAT']
    )

def prepOli(img):
    """
    Prepare Landsat 8/9 imagery by applying cloud mask and renaming bands.
    
    Parameters:
        img (ee.Image): Raw Landsat image
        
    Returns:
        ee.Image: Processed image with cloud masking and standardized band names
    """
    orig = img
    img = maskL8sr(img)  # Apply cloud masking
    img = renameOli(img)  # Rename bands to standardized names

    # Preserve original metadata
    return ee.Image(img.copyProperties(orig, orig.propertyNames()))

def download_landsat(shp_path, start_date, end_date, crs, output_dir):
    """
    Download Landsat 8/9 imagery for the given ROI with cloud masking.
    
    Parameters:
        shp_path (str): Path to shapefile defining the ROI
        start_date (ee.Date): Start date for image search
        end_date (ee.Date): End date for image search
        output_dir (str): Directory to save the downloaded images
        crs (str): Coordinate Reference System to use for the exported images
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert shapefile to Earth Engine geometry
    roi_ee = geemap.shp_to_ee(shp_path)
    roi = roi_ee.geometry()
    
    # Create filter for image collection
    colFilter = ee.Filter.And(
        ee.Filter.bounds(roi_ee), 
        ee.Filter.date(start_date, end_date)
    )
    
    # Get Landsat 8 and 9 collections
    oliCol = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .filter(colFilter) \
            .map(prepOli) 
    oli2Col = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') \
            .filter(colFilter) \
            .map(prepOli)
    
    # Merge Landsat 8 and Landsat 9 collections        
    landsat_collection = oliCol.merge(oli2Col)
    
    # Get the image count
    count = landsat_collection.size().getInfo()
    print(f"Found {count} Landsat images")
    
    if count == 0:
        print("No Landsat images found!")
        return
    else:
        print(f"Processing {count} Landsat images")
        
    # Download images
    images = landsat_collection.toList(count)
    for i in range(count):
        image = ee.Image(images.get(i))
            
        # Clip to ROI
        image_clipped = image.clip(roi_ee)

        # Add RGB composite to the map for visualization
        vis_params = {
            'bands': ['red', 'green', 'blue'],
            'min': 0,
            'max': 0.3,
            'gamma': 1.4
        }
        Map.addLayer(image_clipped, vis_params, f"Landsat RGB {i}")
        
        # Format date for filename
        date_acquired = ee.Date(image.get('system:time_start')).format('YYYYMMdd').getInfo()
        filename = f"Landsat_{date_acquired}_{i}.tif"
        output_path = os.path.join(output_dir, filename)
        
        # Export the image
        print(f"Downloading Landsat image: {filename}")
        geemap.ee_export_image(
            image_clipped, 
            filename=output_path,
            scale=30,  # Landsat 8/9 resolution (30m)
            region=roi,
            file_per_band=True,  # Save each band as a separate file
            crs=crs
        )

#%%
# Initialize Earth Engine and create a map
Map = geemap.Map(basemap='SATELLITE')
Map

#%%
# Path to the input GeoTIFF file (drone orthomosaic)
tiff_path = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/4_Student projects/1_Monika_Kathrine/2_Results/1_DroneImagery/postprocessed/20230608_orthomosaic32022_processed.tif"
shp_path = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/4_Student projects/1_Monika_Kathrine/2_Results/1_DroneImagery/shp/20230608_orthomosaic32022_processed_extent.shp"

# Convert TIFF extent to shapefile or use existing one
roi_gdf = tiff_to_shapefile(tiff_path, shp_path)
crs = roi_gdf.crs
crs_string = f"EPSG:{crs.to_epsg()}" if crs.to_epsg() else None
roi = geemap.shp_to_ee(shp_path)

# Add the ROI to the map and center view
Map.addLayer(roi, {}, "ROI")
Map.centerObject(roi, 16)

#%%
# Set up parameters for satellite imagery search
# Define date range around the drone acquisition date (June 8, 2023)
drone_date = ee.Date.fromYMD(2023, 6, 8)
day_step = 1  # Number of days before/after drone date to search
start_date = drone_date.advance(-day_step, 'day')  # 1 day before
end_date = drone_date.advance(day_step, 'day')  # 1 day after
end_date = end_date.advance(1, 'day')  # Add 1 day to make date range inclusive

# Directory for saving downloaded satellite imagery
output_dir = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/4_Student projects/1_Monika_Kathrine/2_Results/1_DroneImagery/satellite/'

# Download Sentinel-2 imagery after cloud and saturation masking, rescale to 0-1 range
print("\nSearching and downloading Sentinel-2 imagery...")
download_sentinel2(shp_path, start_date, end_date, crs_string, output_dir)

# Download Landsat imagery after cloud and saturation masking, rescale to reflectance
print("\nSearching and downloading Landsat imagery...")
download_landsat(shp_path, start_date, end_date, crs_string, output_dir)

print("\nProcessing complete!")

#%%
# Visualize downloaded satellite imagery
def plot_satellite_rgb(imagery_dir, max_images=3, figsize=(15, 10)):
    """
    Plot RGB composites of downloaded satellite imagery
    
    Parameters:
        imagery_dir (str): Directory containing the satellite imagery
        max_images (int): Maximum number of images to display
        figsize (tuple): Figure size for the plots
    """
    # Find all GeoTIFF files in the directory
    sentinel_files = sorted(glob(os.path.join(imagery_dir, "Sentinel2_*.tif")))
    landsat_files = sorted(glob(os.path.join(imagery_dir, "Landsat_*.tif")))
    
    # Process Sentinel-2 images
    if sentinel_files:
        plt.figure(figsize=figsize)
        plt.suptitle("Sentinel-2 RGB Composites", fontsize=16)
        
        for i, file in enumerate(sentinel_files[:max_images]):
            if i >= max_images:
                break
                
            # Get date from filename
            date_str = os.path.basename(file).split('_')[1]
            
            # For Sentinel-2, we need to find the individual band files 
            # (if file_per_band=True was used)
            base_path = os.path.splitext(file)[0]
            red_path = f"{base_path}_red.tif"
            green_path = f"{base_path}_green.tif"
            blue_path = f"{base_path}_blue.tif"
            
            # Check if individual band files exist
            if os.path.exists(red_path) and os.path.exists(green_path) and os.path.exists(blue_path):
                # Read individual bands
                with rasterio.open(red_path) as src_r, \
                     rasterio.open(green_path) as src_g, \
                     rasterio.open(blue_path) as src_b:
                    
                    red = src_r.read(1)
                    green = src_g.read(1)
                    blue = src_b.read(1)
                    
                    # Stack bands for RGB composite
                    rgb = np.dstack((red, green, blue))
                    
                    # Normalize for display (values were scaled to 0-1 in the download function)
                    rgb_norm = np.clip(rgb * 3.5, 0, 1)  # Adjust multiplier for brightness
                    
                    # Plot
                    plt.subplot(1, len(sentinel_files[:max_images]), i + 1)
                    plt.imshow(rgb_norm)
                    plt.title(f"Sentinel-2 {date_str}")
                    plt.axis('off')
            else:
                # Try to open as a single file with multiple bands
                try:
                    with rasterio.open(file) as src:
                        # Read the first three bands (assuming RGB order)
                        rgb = np.dstack([src.read(i+1) for i in range(3)])
                        
                        # Normalize for display
                        rgb_norm = np.clip(rgb * 3.5, 0, 1)
                        
                        # Plot
                        plt.subplot(1, len(sentinel_files[:max_images]), i + 1)
                        plt.imshow(rgb_norm)
                        plt.title(f"Sentinel-2 {date_str}")
                        plt.axis('off')
                except Exception as e:
                    print(f"Error loading Sentinel-2 image {file}: {e}")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
                
    # Process Landsat images
    if landsat_files:
        plt.figure(figsize=figsize)
        plt.suptitle("Landsat 8/9 RGB Composites", fontsize=16)
        
        for i, file in enumerate(landsat_files[:max_images]):
            if i >= max_images:
                break
                
            # Get date from filename
            date_str = os.path.basename(file).split('_')[1]
            
            # For Landsat, we need to find the individual band files 
            # (if file_per_band=True was used)
            base_path = os.path.splitext(file)[0]
            red_path = f"{base_path}_red.tif"
            green_path = f"{base_path}_green.tif"
            blue_path = f"{base_path}_blue.tif"
            
            # Check if individual band files exist
            if os.path.exists(red_path) and os.path.exists(green_path) and os.path.exists(blue_path):
                # Read individual bands
                with rasterio.open(red_path) as src_r, \
                     rasterio.open(green_path) as src_g, \
                     rasterio.open(blue_path) as src_b:
                    
                    red = src_r.read(1)
                    green = src_g.read(1)
                    blue = src_b.read(1)
                    
                    # Stack bands for RGB composite
                    rgb = np.dstack((red, green, blue))
                    
                    # Normalize for display
                    rgb_norm = np.clip(rgb * 3.5, 0, 1)  # Adjust multiplier for brightness
                    
                    # Plot
                    plt.subplot(1, len(landsat_files[:max_images]), i + 1)
                    plt.imshow(rgb_norm)
                    plt.title(f"Landsat {date_str}")
                    plt.axis('off')
            else:
                # Try to open as a single file with multiple bands
                try:
                    with rasterio.open(file) as src:
                        # Read the first three bands (assuming RGB order)
                        rgb = np.dstack([src.read(i+1) for i in range(3)])
                        
                        # Normalize for display
                        rgb_norm = np.clip(rgb * 3.5, 0, 1)
                        
                        # Plot
                        plt.subplot(1, len(landsat_files[:max_images]), i + 1)
                        plt.imshow(rgb_norm)
                        plt.title(f"Landsat {date_str}")
                        plt.axis('off')
                except Exception as e:
                    print(f"Error loading Landsat image {file}: {e}")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if not sentinel_files and not landsat_files:
        print("No satellite imagery found in the specified directory.")
    
    plt.show()

# Call the function to visualize the downloaded imagery
print("\nVisualizing downloaded satellite imagery...")
satellite_dir = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/4_Student projects/1_Monika_Kathrine/2_Results/1_DroneImagery/satellite/'
plot_satellite_rgb(satellite_dir)


# %%
