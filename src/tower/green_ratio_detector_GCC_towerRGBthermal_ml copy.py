"""
Green Ratio Detector for Vegetation Analysis with Machine Learning Classification

This script processes time-lapse RGB and thermal images to quantify vegetation coverage and
separate different vegetation types (e.g., trees and understory) using machine learning methods.

Main features:
1. Recursively searches for RGB and thermal images in specified directories.
2. Extracts datetime metadata from image EXIF (RGB) and filenames (.mat thermal).
3. Matches each RGB image to the closest-in-time thermal image.
4. Crops images to a region of interest and applies a buffer to avoid edge artifacts.
5. Detects green vegetation using the GCC index and a threshold.
6. Separates green vegetation into two classes using a selected ML method:
    - K-means clustering (default, in LAB color space)
    - Gaussian Mixture Models (GMM)
    - DBSCAN (color + spatial clustering)
    - Spectral clustering
7. Calculates ratios, means, and standard deviations for total green vegetation and each class,
    including corresponding thermal statistics.
8. Saves results to a CSV file and visualizations.
9. Saves classification masks as .mat files for validation.

Dependencies:
- OpenCV (cv2)
- NumPy
- Matplotlib
- scikit-learn
- tqdm
- PIL (Pillow)
- pandas
- seaborn
- cmocean
- scipy.io (for .mat files)

Usage:
- Set 'rgbfolder' and 'thermalfolder' to the directories containing RGB and thermal images.
- Set 'imoutfolder' for output.
- Set 'classification_method' to one of: "kmeans", "gmm", "dbscan", "spectral".
- Run the script to process all images and generate results.

Output:
- Visualizations with original, green-masked, classified, and thermal images.
- CSV file with green ratio and class-specific statistics for all processed images.
- .mat files containing classification masks (green_mask, class1_mask, class2_mask) for validation.

Author: Shunan Feng (shf@ign.ku.dk)
"""
#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import glob
import tqdm
import seaborn as sns
import pandas as pd
import datetime
import cmocean
from PIL import Image
from PIL.ExifTags import TAGS
from scipy.io import loadmat, savemat

# Machine learning imports
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


sns.set_theme(style="darkgrid", font_scale=1.5)

#%%
# Set the classification method to use
classification_method = "kmeans"  # Options: "kmeans", "gmm", "dbscan", "spectral"

#%% load both RGB and thermal images
rgbfolder = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/1_Simon/1_Abisko/6_Tower_Data/Tower RGB images/1 Data/1 Years'
imrgbfiles = []
imrgbfiles.extend(glob.glob(os.path.join(rgbfolder, '**/', '*.JPG'), recursive=True))
imrgbfiles.extend(glob.glob(os.path.join(rgbfolder, '**/', '*.jpg'), recursive=True))
imrgbfiles.extend(glob.glob(os.path.join(rgbfolder, '**/', '*.JPEG'), recursive=True))
imrgbfiles.extend(glob.glob(os.path.join(rgbfolder, '**/', '*.jpeg'), recursive=True))
imrgbfiles.extend(glob.glob(os.path.join(rgbfolder, '**/', '*.png'), recursive=True))
imrgbfiles.extend(glob.glob(os.path.join(rgbfolder, '**/', '*.PNG'), recursive=True))
print(f"Found {len(imrgbfiles)} images in {rgbfolder}")

thermalfolder = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/Tower thermal images/preview/all/registered'
imthermalfiles = []
imthermalfiles.extend(glob.glob(os.path.join(thermalfolder, '**/', '*.mat'), recursive=True))
print(f"Found {len(imthermalfiles)} thermal .mat files in {thermalfolder}")

imoutfolder = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/Tower_RGB_Thermal_Analysis/Data_greenes_thermal_' + classification_method + '_mean'
# Create the output directory if it doesn't exist
if not os.path.exists(imoutfolder):
    os.makedirs(imoutfolder)
#%%
# Create the main results directory if it doesn't exist
results_dir = os.path.join(imoutfolder, 'results')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Create classification masks directory
masks_dir = os.path.join(results_dir, 'classification_masks')
if not os.path.exists(masks_dir):
    os.makedirs(masks_dir)

# Initialize the CSV file with headers
csv_path = os.path.join(results_dir, 'green_ratio_thermal_' + classification_method + '.csv')
with open(csv_path, 'w') as f:
    f.write('filename,datetime,green_ratio,green_mean,green_std,green_norm,class1_ratio,class1_mean,class1_std,class1_norm,class2_ratio,class2_mean,class2_std,class2_norm,class1_temp_mean,class1_temp_std,class2_temp_mean,class2_temp_std,time_diff_sec,method,mask_file\n')
#%%
def get_image_rgb_datetime(image_path):
    """
    Extract the datetime when the image was taken from EXIF metadata.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        str: Datetime string in ISO format (YYYY-MM-DD HH:MM:SS) or None if not available
    """
    try:
        with Image.open(image_path) as img:
            # Extract EXIF data
            exifdata = img._getexif()
            
            if exifdata is None:
                return None
                
            # Look for DateTimeOriginal (36867) or DateTime (306) tag
            datetime_taken = None
            for tag_id, value in exifdata.items():
                tag = TAGS.get(tag_id, tag_id)
                
                # Check for different date time tags
                if tag == 'DateTimeOriginal' or tag == 'DateTime':
                    datetime_taken = value
                    break
            
            if datetime_taken:
                # Convert from EXIF format (YYYY:MM:DD HH:MM:SS) to ISO format
                try:
                    # Parse the datetime
                    dt = datetime.datetime.strptime(datetime_taken, '%Y:%m:%d %H:%M:%S')
                    return dt.strftime('%Y-%m-%d %H:%M:%S')
                except ValueError:
                    # In case the format is unexpected
                    return datetime_taken
                    
            return None
    except Exception as e:
        print(f"Error extracting metadata from {image_path}: {e}")
        return None

#%%
def get_image_thermal_datetime(image_path):
    """
    Extract the datetime when the thermal image was taken from .mat filename.
    
    Args:
        image_path: Path to the .mat file   
    Returns:
        str: Datetime string in ISO format (YYYY-MM-DD HH:MM:SS) or None if not available
    """
    try:
        # Extract filename without extension
        base_name = os.path.basename(image_path)
        name, _ = os.path.splitext(base_name)
        
        # Assuming the filename contains datetime in format 'prefix_YYYY-MM-DD_HH.MM.SS.mat'
        parts = name.split('_')
        
        try :
            date_part = parts[-2]  # 'YYYY-MM-DD'
            time_part = parts[-1]  # 'HH.MM.SS'
            time_part = time_part.replace('.', ':')  # Replace '.' with ':'
            datetime_str = f"{date_part} {time_part}"
            
            # Validate and return in ISO format
            dt = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            return None
    except Exception as e:
        print(f"Error extracting datetime from {image_path}: {e}")
        return None
#%% 
def get_image_position(image_path, image_type):
    """
    Extract the position (North-facing or West-facing) from the image filename.
    
    Args:
        image_name: Name of the image file
    Returns:
        str: 'North-facing', 'West-facing', or 'unknown'
    """
    if image_type == 'rgb':
        img_position = os.path.normpath(image_path).split(os.sep)[-3]
        return img_position
    elif image_type == 'thermal':
        base_name = os.path.basename(image_path)
        name, _ = os.path.splitext(base_name)
        parts = name.split('_')
        img_position = parts[0]
        return img_position
    else:
        return 'unknown'

#%% mask and crop the image to the area of interest
def mask_and_crop_image(img):
    """
    Crop the image to remove mismatches and edge effects.

    - Crops the image to the first 800 rows to match the region of interest.
    - Applies a 50-pixel buffer on all sides to avoid edge artifacts.

    Args:
        img (np.ndarray): Input image (BGR or grayscale).

    Returns:
        np.ndarray: Cropped image.
    """
    crop_top = 800
    buffer = 50

    # Crop to first 800 rows
    img = img[:crop_top]

    # Apply buffer to all sides
    if img.ndim == 3:
        img = img[buffer:-buffer, buffer:-buffer, :]
    else:
        img = img[buffer:-buffer, buffer:-buffer]

    return img

#%%    
def quantify_vegetation_kmeans(img_rgb, img_thermal):
    """
    Quantifies green vegetation using K-means clustering.
    
    This method first identifies green pixels using GCC index, then separates them
    into two classes (likely trees and understory) using K-means clustering in LAB color space.
    
    Args:
        img: The input image (BGR format)

    Returns:
        tuple: Overall green metrics (ratio, mean, std, norm_greenness),
               green mask, 
               class1 metrics (ratio, mean, std, norm_greenness), 
               class2 metrics (ratio, mean, std, norm_greenness), 
               class visualization
    """
    try:
        # Split the image into its BGR channels
        b, g, r = cv2.split(img_rgb)
        
        # Convert to float to avoid integer division
        b = b.astype(float)
        g = g.astype(float)
        r = r.astype(float)
        
        # Calculate the Greenness index G/(R+G+B)
        # Note: Adding a small value to prevent division by zero
        greenness = g / (r + g + b + 1e-10)

        # Create a binary mask using a threshold (adjust as needed)
        threshold = 0.37 # 0.38 +- 0.01
        green_mask = (greenness > threshold).astype(np.uint8) * 255
        
        # Count green pixels and calculate ratio
        green_pixels = np.sum(green_mask > 0)
        total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
        green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0
        
        # Calculate mean and std greenness for all green pixels
        green_pixels_mask = green_mask > 0
        if green_pixels > 0:
            mean_greenness = np.mean(greenness[green_pixels_mask])
            std_greenness = np.std(greenness[green_pixels_mask])
            norm_greenness = np.sum(greenness[green_pixels_mask]) / green_pixels
            mean_temperature = np.nanmean(img_thermal[green_pixels_mask])
            std_temperature = np.nanstd(img_thermal[green_pixels_mask])
        else:
            mean_greenness = 0
            std_greenness = 0
            norm_greenness = 0
            mean_temperature = np.nan
            std_temperature = np.nan

        # Store overall green metrics as a dictionary
        green_metrics = {
            'ratio': green_ratio,
            'mean': mean_greenness,
            'std': std_greenness,
            'norm_greenness': norm_greenness,
            'mean_temperature': mean_temperature,
            'std_temperature': std_temperature
        }
        
        # ENHANCED CLASSIFICATION: Using K-means for vegetation class separation
        
        # Step 1: Create a masked green-only image
        masked_green = cv2.bitwise_and(img_rgb, img_rgb, mask=(green_mask // 255).astype(np.uint8))
        
        # Step 2: Prepare data for K-means clustering - only include green pixels
        non_zero_mask = np.any(masked_green != 0, axis=2)
        
        # Initialize class metrics
        class1_metrics = {'ratio': 0, 'mean': 0, 'std': 0, 'norm_greenness': 0}
        class2_metrics = {'ratio': 0, 'mean': 0, 'std': 0, 'norm_greenness': 0}
        visualization = np.zeros_like(img_rgb)
        
        if np.sum(non_zero_mask) > 0:  # Check if there are green pixels
            # Extract features for clustering (using multiple channels for better separation)
            # Convert to LAB color space which is better for color-based segmentation
            lab_image = cv2.cvtColor(masked_green, cv2.COLOR_BGR2LAB)
            
            # Reshape to a list of pixels with features
            pixels = lab_image[non_zero_mask].reshape(-1, 3).astype(np.float32)
            
            # Apply K-means clustering (K=2 for trees and understory)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Create masks for each class
            # First, create a full-size labels array initialized with -1 (no class)
            full_labels = np.full(img_rgb.shape[:2], -1, dtype=int)
            # Then, set the labels only for green pixels
            full_labels[non_zero_mask] = labels.flatten()
            
            # Determine which cluster is likely understory (darker) and which is trees (brighter)
            if centers[0][0] < centers[1][0]:  # Lower L value = darker
                class1_mask = (full_labels == 0)  # Understory (darker)
                class2_mask = (full_labels == 1)  # Trees (brighter)
            else:
                class1_mask = (full_labels == 1)  # Understory (darker)
                class2_mask = (full_labels == 0)  # Trees (brighter)
            
            # Calculate individual class ratios
            class1_pixels = np.sum(class1_mask)
            class2_pixels = np.sum(class2_mask)
            
            class1_ratio = class1_pixels / total_pixels if total_pixels > 0 else 0
            class2_ratio = class2_pixels / total_pixels if total_pixels > 0 else 0
            
            # Calculate mean, std, and normalized greenness, and mean, std temperature for each class
            if class1_pixels > 0:
                class1_mean = np.mean(greenness[class1_mask])
                class1_std = np.std(greenness[class1_mask])
                class1_norm_greenness = np.sum(greenness[class1_mask]) / class1_pixels
                class1_mean_temperature = np.nanmean(img_thermal[class1_mask])
                class1_std_temperature = np.nanstd(img_thermal[class1_mask])
            else:
                class1_mean = 0
                class1_std = 0
                class1_norm_greenness = 0
                class1_mean_temperature = np.nan
                class1_std_temperature = np.nan

            if class2_pixels > 0:
                class2_mean = np.mean(greenness[class2_mask])
                class2_std = np.std(greenness[class2_mask])
                class2_norm_greenness = np.sum(greenness[class2_mask]) / class2_pixels
                class2_mean_temperature = np.nanmean(img_thermal[class2_mask])
                class2_std_temperature = np.nanstd(img_thermal[class2_mask])
            else:
                class2_mean = 0
                class2_std = 0
                class2_norm_greenness = 0
                class2_mean_temperature = np.nan
                class2_std_temperature = np.nan
                
            # Store class metrics
            class1_metrics = {
                'ratio': class1_ratio,
                'mean': class1_mean,
                'std': class1_std,
                'norm_greenness': class1_norm_greenness,
                'mean_temperature': class1_mean_temperature,
                'std_temperature': class1_std_temperature
            }
            
            class2_metrics = {
                'ratio': class2_ratio,
                'mean': class2_mean,
                'std': class2_std,
                'norm_greenness': class2_norm_greenness,
                'mean_temperature': class2_mean_temperature,
                'std_temperature': class2_std_temperature
            }
            
            # Create a visualization of the two classes
            visualization = np.zeros_like(img_rgb)
            # Class 1 - understory (shown in blue)
            visualization[class1_mask] = [255, 0, 0]
            # Class 2 - trees (shown in green) 
            visualization[class2_mask] = [0, 255, 0]
        
        return green_metrics, green_mask, class1_metrics, class2_metrics, visualization

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None, None

def quantify_vegetation_gmm(img_rgb, img_thermal):
    """
    Quantifies vegetation using Gaussian Mixture Models (GMM).
    Uses LAB color space for clustering, similar to kmeans, and includes thermal stats.
    """
    try:
        # Split the image into its BGR channels
        b, g, r = cv2.split(img_rgb)
        b = b.astype(float)
        g = g.astype(float)
        r = r.astype(float)
        greenness = g / (r + g + b + 1e-10)
        threshold = 0.37
        green_mask = (greenness > threshold).astype(np.uint8) * 255

        green_pixels = np.sum(green_mask > 0)
        total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
        green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0

        green_pixels_mask = green_mask > 0
        if green_pixels > 0:
            mean_greenness = np.mean(greenness[green_pixels_mask])
            std_greenness = np.std(greenness[green_pixels_mask])
            norm_greenness = np.sum(greenness[green_pixels_mask]) / green_pixels
            mean_temperature = np.nanmean(img_thermal[green_pixels_mask])
            std_temperature = np.nanstd(img_thermal[green_pixels_mask])
        else:
            mean_greenness = 0
            std_greenness = 0
            norm_greenness = 0
            mean_temperature = np.nan
            std_temperature = np.nan

        green_metrics = {
            'ratio': green_ratio,
            'mean': mean_greenness,
            'std': std_greenness,
            'norm_greenness': norm_greenness,
            'mean_temperature': mean_temperature,
            'std_temperature': std_temperature
        }

        class1_metrics = {'ratio': 0, 'mean': 0, 'std': 0, 'norm_greenness': 0, 'mean_temperature': np.nan, 'std_temperature': np.nan}
        class2_metrics = {'ratio': 0, 'mean': 0, 'std': 0, 'norm_greenness': 0, 'mean_temperature': np.nan, 'std_temperature': np.nan}
        visualization = np.zeros_like(img_rgb)

        non_zero_mask = green_mask > 0
        if np.sum(non_zero_mask) > 0:
            masked_green = cv2.bitwise_and(img_rgb, img_rgb, mask=(green_mask // 255).astype(np.uint8))
            lab_image = cv2.cvtColor(masked_green, cv2.COLOR_BGR2LAB)
            pixels = lab_image[non_zero_mask].reshape(-1, 3).astype(np.float32)

            gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
            labels = gmm.fit_predict(pixels)

            full_labels = np.full(img_rgb.shape[:2], -1, dtype=int)
            full_labels[non_zero_mask] = labels

            if gmm.means_[0][0] < gmm.means_[1][0]:
                class1_mask = (full_labels == 0)
                class2_mask = (full_labels == 1)
            else:
                class1_mask = (full_labels == 1)
                class2_mask = (full_labels == 0)

            class1_pixels = np.sum(class1_mask)
            class2_pixels = np.sum(class2_mask)
            class1_ratio = class1_pixels / total_pixels if total_pixels > 0 else 0
            class2_ratio = class2_pixels / total_pixels if total_pixels > 0 else 0

            if class1_pixels > 0:
                class1_mean = np.mean(greenness[class1_mask])
                class1_std = np.std(greenness[class1_mask])
                class1_norm_greenness = np.sum(greenness[class1_mask]) / class1_pixels
                class1_mean_temperature = np.nanmean(img_thermal[class1_mask])
                class1_std_temperature = np.nanstd(img_thermal[class1_mask])
            else:
                class1_mean = 0
                class1_std = 0
                class1_norm_greenness = 0
                class1_mean_temperature = np.nan
                class1_std_temperature = np.nan

            if class2_pixels > 0:
                class2_mean = np.mean(greenness[class2_mask])
                class2_std = np.std(greenness[class2_mask])
                class2_norm_greenness = np.sum(greenness[class2_mask]) / class2_pixels
                class2_mean_temperature = np.nanmean(img_thermal[class2_mask])
                class2_std_temperature = np.nanstd(img_thermal[class2_mask])
            else:
                class2_mean = 0
                class2_std = 0
                class2_norm_greenness = 0
                class2_mean_temperature = np.nan
                class2_std_temperature = np.nan

            class1_metrics = {
                'ratio': class1_ratio,
                'mean': class1_mean,
                'std': class1_std,
                'norm_greenness': class1_norm_greenness,
                'mean_temperature': class1_mean_temperature,
                'std_temperature': class1_std_temperature
            }
            class2_metrics = {
                'ratio': class2_ratio,
                'mean': class2_mean,
                'std': class2_std,
                'norm_greenness': class2_norm_greenness,
                'mean_temperature': class2_mean_temperature,
                'std_temperature': class2_std_temperature
            }

            visualization = np.zeros_like(img_rgb)
            visualization[class1_mask] = [255, 0, 0]
            visualization[class2_mask] = [0, 255, 0]

        return green_metrics, green_mask, class1_metrics, class2_metrics, visualization

    except Exception as e:
        print(f"An error occurred in GMM clustering: {e}")
        return None, None, None, None, None

def quantify_vegetation_dbscan(img_rgb, img_thermal):
    """
    Quantifies vegetation using DBSCAN clustering.
    Uses LAB color + spatial features, and includes thermal stats.
    """
    try:
        b, g, r = cv2.split(img_rgb)
        b, g, r = b.astype(float), g.astype(float), r.astype(float)
        greenness = g / (r + g + b + 1e-10)
        threshold = 0.37
        green_mask = (greenness > threshold).astype(np.uint8) * 255

        green_pixels = np.sum(green_mask > 0)
        total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
        green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0

        green_pixels_mask = green_mask > 0
        if green_pixels > 0:
            mean_greenness = np.mean(greenness[green_pixels_mask])
            std_greenness = np.std(greenness[green_pixels_mask])
            norm_greenness = np.sum(greenness[green_pixels_mask]) / green_pixels
            mean_temperature = np.nanmean(img_thermal[green_pixels_mask])
            std_temperature = np.nanstd(img_thermal[green_pixels_mask])
        else:
            mean_greenness = 0
            std_greenness = 0
            norm_greenness = 0
            mean_temperature = np.nan
            std_temperature = np.nan

        green_metrics = {
            'ratio': green_ratio,
            'mean': mean_greenness,
            'std': std_greenness,
            'norm_greenness': norm_greenness,
            'mean_temperature': mean_temperature,
            'std_temperature': std_temperature
        }

        class1_metrics = {'ratio': 0, 'mean': 0, 'std': 0, 'norm_greenness': 0, 'mean_temperature': np.nan, 'std_temperature': np.nan}
        class2_metrics = {'ratio': 0, 'mean': 0, 'std': 0, 'norm_greenness': 0, 'mean_temperature': np.nan, 'std_temperature': np.nan}
        visualization = np.zeros_like(img_rgb)

        non_zero_mask = green_mask > 0
        if np.sum(non_zero_mask) > 100:
            y_coords, x_coords = np.where(non_zero_mask)
            masked_green = cv2.bitwise_and(img_rgb, img_rgb, mask=(green_mask // 255).astype(np.uint8))
            lab_image = cv2.cvtColor(masked_green, cv2.COLOR_BGR2LAB)
            spatial_weight = 0.01
            features = np.column_stack([
                lab_image[non_zero_mask],
                x_coords * spatial_weight,
                y_coords * spatial_weight
            ])
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            dbscan = DBSCAN(eps=0.5, min_samples=10)
            labels = dbscan.fit_predict(scaled_features)
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels >= 0]
            if len(unique_labels) >= 2:
                counts = np.array([np.sum(labels == label) for label in unique_labels])
                largest_indices = np.argsort(counts)[-2:]
                class1_label = unique_labels[largest_indices[1]]
                class2_label = unique_labels[largest_indices[0]]
                class1_indices = np.where(labels == class1_label)[0]
                class2_indices = np.where(labels == class2_label)[0]
                y1, x1 = y_coords[class1_indices], x_coords[class1_indices]
                y2, x2 = y_coords[class2_indices], x_coords[class2_indices]
                class1_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
                class2_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
                class1_mask[y1, x1] = True
                class2_mask[y2, x2] = True

                class1_pixels = np.sum(class1_mask)
                class2_pixels = np.sum(class2_mask)
                class1_ratio = class1_pixels / total_pixels if total_pixels > 0 else 0
                class2_ratio = class2_pixels / total_pixels if total_pixels > 0 else 0

                if class1_pixels > 0:
                    class1_mean = np.mean(greenness[class1_mask])
                    class1_std = np.std(greenness[class1_mask])
                    class1_norm_greenness = np.sum(greenness[class1_mask]) / class1_pixels
                    class1_mean_temperature = np.nanmean(img_thermal[class1_mask])
                    class1_std_temperature = np.nanstd(img_thermal[class1_mask])
                else:
                    class1_mean = 0
                    class1_std = 0
                    class1_norm_greenness = 0
                    class1_mean_temperature = np.nan
                    class1_std_temperature = np.nan

                if class2_pixels > 0:
                    class2_mean = np.mean(greenness[class2_mask])
                    class2_std = np.std(greenness[class2_mask])
                    class2_norm_greenness = np.sum(greenness[class2_mask]) / class2_pixels
                    class2_mean_temperature = np.nanmean(img_thermal[class2_mask])
                    class2_std_temperature = np.nanstd(img_thermal[class2_mask])
                else:
                    class2_mean = 0
                    class2_std = 0
                    class2_norm_greenness = 0
                    class2_mean_temperature = np.nan
                    class2_std_temperature = np.nan

                class1_metrics = {
                    'ratio': class1_ratio,
                    'mean': class1_mean,
                    'std': class1_std,
                    'norm_greenness': class1_norm_greenness,
                    'mean_temperature': class1_mean_temperature,
                    'std_temperature': class1_std_temperature
                }
                class2_metrics = {
                    'ratio': class2_ratio,
                    'mean': class2_mean,
                    'std': class2_std,
                    'norm_greenness': class2_norm_greenness,
                    'mean_temperature': class2_mean_temperature,
                    'std_temperature': class2_std_temperature
                }
                visualization = np.zeros_like(img_rgb)
                visualization[class1_mask] = [255, 0, 0]
                visualization[class2_mask] = [0, 255, 0]
            else:
                class1_metrics = {
                    'ratio': green_ratio,
                    'mean': mean_greenness,
                    'std': std_greenness,
                    'norm_greenness': norm_greenness,
                    'mean_temperature': mean_temperature,
                    'std_temperature': std_temperature
                }
                visualization = np.zeros_like(img_rgb)
                visualization[non_zero_mask] = [255, 0, 0]

        return green_metrics, green_mask, class1_metrics, class2_metrics, visualization

    except Exception as e:
        print(f"An error occurred in DBSCAN clustering: {e}")
        return None, None, None, None, None

def quantify_vegetation_spectral(img_rgb, img_thermal):
    """
    Quantifies vegetation using spectral clustering.
    Uses LAB color space for clustering, includes thermal stats.
    """
    try:
        b, g, r = cv2.split(img_rgb)
        b, g, r = b.astype(float), g.astype(float), r.astype(float)
        greenness = g / (r + g + b + 1e-10)
        threshold = 0.37
        green_mask = (greenness > threshold).astype(np.uint8) * 255

        green_pixels = np.sum(green_mask > 0)
        total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
        green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0

        green_pixels_mask = green_mask > 0
        if green_pixels > 0:
            mean_greenness = np.mean(greenness[green_pixels_mask])
            std_greenness = np.std(greenness[green_pixels_mask])
            norm_greenness = np.sum(greenness[green_pixels_mask]) / green_pixels
            mean_temperature = np.nanmean(img_thermal[green_pixels_mask])
            std_temperature = np.nanstd(img_thermal[green_pixels_mask])
        else:
            mean_greenness = 0
            std_greenness = 0
            norm_greenness = 0
            mean_temperature = np.nan
            std_temperature = np.nan

        green_metrics = {
            'ratio': green_ratio,
            'mean': mean_greenness,
            'std': std_greenness,
            'norm_greenness': norm_greenness,
            'mean_temperature': mean_temperature,
            'std_temperature': std_temperature
        }

        class1_metrics = {'ratio': 0, 'mean': 0, 'std': 0, 'norm_greenness': 0, 'mean_temperature': np.nan, 'std_temperature': np.nan}
        class2_metrics = {'ratio': 0, 'mean': 0, 'std': 0, 'norm_greenness': 0, 'mean_temperature': np.nan, 'std_temperature': np.nan}
        visualization = np.zeros_like(img_rgb)

        non_zero_mask = green_mask > 0
        if np.sum(non_zero_mask) > 100:
            masked_green = cv2.bitwise_and(img_rgb, img_rgb, mask=(green_mask // 255).astype(np.uint8))
            lab_image = cv2.cvtColor(masked_green, cv2.COLOR_BGR2LAB)
            features = lab_image[non_zero_mask].reshape(-1, 3)
            max_samples = 10000
            if len(features) > max_samples:
                indices = np.random.choice(len(features), max_samples, replace=False)
                sampled_features = features[indices]
                spectral = SpectralClustering(n_clusters=2, random_state=42, assign_labels='kmeans')
                sampled_labels = spectral.fit_predict(sampled_features)
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(sampled_features, sampled_labels)
                labels = knn.predict(features)
            else:
                spectral = SpectralClustering(n_clusters=2, random_state=42, assign_labels='kmeans')
                labels = spectral.fit_predict(features)

            class1_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
            class2_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
            y_indices, x_indices = np.where(non_zero_mask)
            class1_indices = np.where(labels == 0)[0]
            class2_indices = np.where(labels == 1)[0]
            class1_mask[y_indices[class1_indices], x_indices[class1_indices]] = True
            class2_mask[y_indices[class2_indices], x_indices[class2_indices]] = True

            class1_pixels = np.sum(class1_mask)
            class2_pixels = np.sum(class2_mask)
            class1_ratio = class1_pixels / total_pixels if total_pixels > 0 else 0
            class2_ratio = class2_pixels / total_pixels if total_pixels > 0 else 0

            if class1_pixels > 0:
                class1_mean = np.mean(greenness[class1_mask])
                class1_std = np.std(greenness[class1_mask])
                class1_norm_greenness = np.sum(greenness[class1_mask]) / class1_pixels
                class1_mean_temperature = np.nanmean(img_thermal[class1_mask])
                class1_std_temperature = np.nanstd(img_thermal[class1_mask])
            else:
                class1_mean = 0
                class1_std = 0
                class1_norm_greenness = 0
                class1_mean_temperature = np.nan
                class1_std_temperature = np.nan

            if class2_pixels > 0:
                class2_mean = np.mean(greenness[class2_mask])
                class2_std = np.std(greenness[class2_mask])
                class2_norm_greenness = np.sum(greenness[class2_mask]) / class2_pixels
                class2_mean_temperature = np.nanmean(img_thermal[class2_mask])
                class2_std_temperature = np.nanstd(img_thermal[class2_mask])
            else:
                class2_mean = 0
                class2_std = 0
                class2_norm_greenness = 0
                class2_mean_temperature = np.nan
                class2_std_temperature = np.nan

            class1_metrics = {
                'ratio': class1_ratio,
                'mean': class1_mean,
                'std': class1_std,
                'norm_greenness': class1_norm_greenness,
                'mean_temperature': class1_mean_temperature,
                'std_temperature': class1_std_temperature
            }
            class2_metrics = {
                'ratio': class2_ratio,
                'mean': class2_mean,
                'std': class2_std,
                'norm_greenness': class2_norm_greenness,
                'mean_temperature': class2_mean_temperature,
                'std_temperature': class2_std_temperature
            }
            visualization = np.zeros_like(img_rgb)
            visualization[class1_mask] = [255, 0, 0]
            visualization[class2_mask] = [0, 255, 0]

        return green_metrics, green_mask, class1_metrics, class2_metrics, visualization

    except Exception as e:
        print(f"An error occurred in spectral clustering: {e}")
        return None, None, None, None, None
    
def quantify_vegetation(img_rgb, img_thermal, method="kmeans"):
    """
    Quantifies vegetation using various clustering methods.
    
    Args:
        img_rgb: The input RGB image (BGR format)
        img_thermal: The input thermal image (grayscale format)
        method: Clustering method to use ('kmeans', 'gmm', 'dbscan', 'spectral')
        
    Returns:
        tuple: Overall green metrics (ratio, mean, std, norm_greenness),
               green mask, 
               class1 metrics (ratio, mean, std, norm_greenness), 
               class2 metrics (ratio, mean, std, norm_greenness), 
               class visualization
    """
    if method == "kmeans":
        print("Using K-means clustering")
        return quantify_vegetation_kmeans(img_rgb, img_thermal)
    elif method == "gmm":
        print("Using Gaussian Mixture Models")
        return quantify_vegetation_gmm(img_rgb, img_thermal)
    elif method == "dbscan":
        print("Using DBSCAN clustering")
        return quantify_vegetation_dbscan(img_rgb, img_thermal)
    elif method == "spectral":
        print("Using Spectral clustering")
        return quantify_vegetation_spectral(img_rgb, img_thermal)
    else:
        print(f"Unknown method '{method}', using kmeans instead")
        return quantify_vegetation_kmeans(img_rgb, img_thermal)

#%% Main processing loop
# Get the list of image datetime for thermal images
imthermalfiles = pd.DataFrame({'file': imthermalfiles})
imthermalfiles['datetime'] = imthermalfiles['file'].apply(get_image_thermal_datetime)
imthermalfiles['position'] = imthermalfiles['file'].apply(get_image_position, image_type='thermal')

# temporary step to process only "West-facing" images
imrgbfiles = [f for f in imrgbfiles if 'West' in f]
# Process each image and write results immediately to CSV
for i in tqdm.tqdm(imrgbfiles, desc="Processing images"):
    img_rgb = cv2.imread(i)
    if img_rgb is None:
        print(f"Could not read image: {i}")
        continue
        
    print("processing: ", i)

    # Get RGB image datetime from EXIF metadata
    img_rgb_datetime = get_image_rgb_datetime(i)
    datetime_str = img_rgb_datetime if img_rgb_datetime else "NA"
    print(f"Image datetime: {datetime_str}")

    # determine if the rgb image is north-facing or west-facing
    img_rgb_position = get_image_position(i, image_type='rgb')
    # filter thermal images to only those with the same position
    imthermalfiles_filtered = imthermalfiles[imthermalfiles['position'] == img_rgb_position]
   
    # Find the thermal image that is closest in time to the RGB image
    img_rgb_datetime_dt = pd.to_datetime(img_rgb_datetime)
    imthermalfiles_filtered['time_diff'] = imthermalfiles_filtered['datetime'].apply(lambda x: abs((x - img_rgb_datetime_dt).total_seconds()))
    closest_thermal = imthermalfiles_filtered.loc[imthermalfiles_filtered['time_diff'].idxmin()]
    thermal_file = closest_thermal['file']
    time_diff_seconds = closest_thermal['time_diff']
    print(f"Closest thermal image: {thermal_file} (time difference: {time_diff_seconds} seconds)")
    
    # Load the thermal image (MAT file)
    thermal_mat = loadmat(thermal_file)
    if 'thermal_image_registered' in thermal_mat:
        img_thermal = thermal_mat['thermal_image_registered']
    elif 'thermal_image' in thermal_mat:
        img_thermal = thermal_mat['thermal_image']
    else:
        print(f"No thermal_image variable found in {thermal_file}")
        continue

    # apply cropping
    img_rgb = mask_and_crop_image(img=img_rgb)
    img_thermal = mask_and_crop_image(img=img_thermal)
    img_thermal = np.double(img_thermal) / 100
    img_thermal[img_thermal == 0] = np.nan  # Set invalid values to NaN
    img_thermal = img_thermal - 273.15  # Convert from Kelvin to Celsius

    # Quantify vegetation within the whole image using the selected method
    green_metrics, green_mask, class1_metrics, class2_metrics, class_vis = quantify_vegetation(img_rgb, img_thermal, method=classification_method)

    if green_metrics is not None:
        print(f"The green pixel ratio is: {green_metrics['ratio']:.4f}")
        print(f"Green mean: {green_metrics['mean']:.4f}, std: {green_metrics['std']:.4f}, normalized: {green_metrics['norm_greenness']:.4f}")
        print(f"Class 1 ratio: {class1_metrics['ratio']:.4f}, mean: {class1_metrics['mean']:.4f}, std: {class1_metrics['std']:.4f}")
        print(f"Class 2 ratio: {class2_metrics['ratio']:.4f}, mean: {class2_metrics['mean']:.4f}, std: {class2_metrics['std']:.4f}")
        print(f"Class 1 mean temperature: {class1_metrics['mean_temperature']:.2f}°C, std: {class1_metrics['std_temperature']:.2f}")
        print(f"Class 2 mean temperature: {class2_metrics['mean_temperature']:.2f}°C, std: {class2_metrics['std_temperature']:.2f}")
        
        # Save classification masks as .mat file
        base_filename = datetime_str.replace(':', '-').replace(' ', '_') if datetime_str != "NA" else os.path.splitext(os.path.basename(i))[0]
        mask_filename = f"{base_filename}_{classification_method}_masks.mat"
        mask_filepath = os.path.join(masks_dir, mask_filename)
        
        # Extract class masks from visualization
        class1_mask = (class_vis[:, :, 0] == 255) & (class_vis[:, :, 1] == 0) & (class_vis[:, :, 2] == 0)  # Blue pixels
        class2_mask = (class_vis[:, :, 0] == 0) & (class_vis[:, :, 1] == 255) & (class_vis[:, :, 2] == 0)  # Green pixels
        
        # Save masks with metadata
        savemat(mask_filepath, {
            'green_mask': green_mask.astype(np.uint8),
            'class1_mask': class1_mask.astype(np.uint8),
            'class2_mask': class2_mask.astype(np.uint8),
            'metadata': {
                'datetime': datetime_str,
                'method': classification_method,
                'rgb_file': i,
                'thermal_file': thermal_file,
                'green_ratio': green_metrics['ratio'],
                'class1_ratio': class1_metrics['ratio'],
                'class2_ratio': class2_metrics['ratio'],
                'class1_temp_mean': class1_metrics['mean_temperature'],
                'class2_temp_mean': class2_metrics['mean_temperature']
            }
        })
        print(f"Saved classification masks to: {mask_filepath}")
        
        # Write result to CSV immediately
        with open(csv_path, 'a') as f:
            f.write(f'{i},{datetime_str},'
                    f'{green_metrics["ratio"]},{green_metrics["mean"]},{green_metrics["std"]},{green_metrics["norm_greenness"]},'
                    f'{class1_metrics["ratio"]},{class1_metrics["mean"]},{class1_metrics["std"]},{class1_metrics["norm_greenness"]},'
                    f'{class2_metrics["ratio"]},{class2_metrics["mean"]},{class2_metrics["std"]},{class2_metrics["norm_greenness"]},'
                    f'{class1_metrics["mean_temperature"]},{class1_metrics["std_temperature"]},{class2_metrics["mean_temperature"]},{class2_metrics["std_temperature"]},'
                    f'{time_diff_seconds},{classification_method},{mask_filename}\n')

        # Apply the green mask to the image
        masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=green_mask // 255)

        # Display the original image, the masked image, and vegetation classes
        # Convert the images to RGB format for matplotlib
        img_rgb_vis = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        masked_img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
        class_vis_rgb = cv2.cvtColor(class_vis, cv2.COLOR_BGR2RGB)

        # Create a figure and axes
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Display the original image
        axes[0].imshow(img_rgb_vis)
        title_text = f"Original ({datetime_str})" if datetime_str != "NA" else "Original"
        axes[0].set_title(title_text)
        axes[0].axis('off')

        # Display the masked image
        axes[1].imshow(masked_img_rgb)
        axes[1].set_title(f"Green Masked (GR={green_metrics['ratio']:.2f}, Norm={green_metrics['norm_greenness']:.2f})")
        axes[1].axis('off')
        
        # Display the vegetation classes
        axes[2].imshow(class_vis_rgb)
        axes[2].set_title(
            f"Vegetation Classes ({classification_method})\n"
            f"class1={class1_metrics['ratio']:.2f} (Temp={class1_metrics['mean_temperature']:.2f}±{class1_metrics['std_temperature']:.2f})\n"
            f"class2={class2_metrics['ratio']:.2f} (Temp={class2_metrics['mean_temperature']:.2f}±{class2_metrics['std_temperature']:.2f})"
        )
        axes[2].axis('off')
        
        # Create custom legend elements
        legend_elements = [
            Patch(facecolor='blue', edgecolor='black', label='Class 1'),
            Patch(facecolor='green', edgecolor='black', label='Class 2')
        ]
        
        # Add legend
        axes[2].legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, -0.3), frameon=True, 
              facecolor='white', edgecolor='black')
        
        # Display the thermal image with colorbar
        im = axes[3].imshow(img_thermal, cmap=cmocean.cm.thermal)
        axes[3].set_title('Thermal Image')
        axes[3].axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        cbar.set_label('Temperature (°C)', rotation=270, labelpad=15)

        plt.tight_layout()

        # Create output directory structure that mirrors input
        rel_path = os.path.relpath(os.path.dirname(i), rgbfolder)
        output_dir = os.path.join(imoutfolder, rel_path)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save the figure
        base_filename_plot = datetime_str.replace(':', '-') if datetime_str != "NA" else os.path.splitext(os.path.basename(i))[0]
        extension = os.path.splitext(os.path.basename(i))[1]
        output_path = os.path.join(output_dir, f"{base_filename_plot}_green_masked{extension}")
        fig.savefig(output_path)
        plt.close(fig)

    else:
        print(f"Vegetation quantification failed for {i}")
        # Write failure to CSV
        with open(csv_path, 'a') as f:
            f.write(f'{i},{datetime_str},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,{classification_method},NA\n')

print(f"Done! Results saved to {csv_path}")
print(f"Classification masks saved to {masks_dir}")
