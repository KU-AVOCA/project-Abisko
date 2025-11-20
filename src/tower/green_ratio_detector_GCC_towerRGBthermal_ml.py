"""
Green Ratio Detector for Single RGB Image with Median of Nearest Thermal Images

This script processes a single RGB image and finds the 5 nearest thermal images,
computes their median, and creates a single visualization.

Author: Shunan Feng (shf@ign.ku.dk)
"""
#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import glob
import datetime
import cmocean
from PIL import Image
from PIL.ExifTags import TAGS
from scipy.io import loadmat
import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
#%%
# ============================================================================
# CONFIGURATION - Specify your file paths here
# ============================================================================
# RGB_IMAGE_PATH = "/data/shunan/data/KU/rgbimages/West-facing/2021/IMG_2270.JPG" # 2021-07-01 12:59:59
# RGB_IMAGE_PATH = "/data/shunan/data/KU/rgbimages/West-facing/2022/IMG_4984.JPG" # 2022-07-04 12:42:23
RGB_IMAGE_PATH = "/data/shunan/data/KU/rgbimages/West-facing/2023/IMG_7402.JPG" # 2023-07-04 12:59:42
THERMAL_FOLDER = "/data_3/shunan_2/KU/registeredMatImages/"
SUPERVISED_MODEL_PATH = "/data/shunan/github/project-Abisko/src/tower/vegetation_classifier.pkl"
OUTPUT_DIR = "/data/shunan/github/project-Abisko/print/tower/"
sns.set_theme(style="whitegrid", font_scale=1.5)

# Number of nearest thermal images to process
NUM_NEAREST_THERMAL = 5

# Color scheme for visualization (RGB format)
BIRCH_COLOR = (115, 172, 49)  # #73ac31 in RGB
UNDERSTORY_COLOR = (205, 205, 205)  # #cdcdcd in RGB
NON_GREEN_COLOR = (0, 0, 0)  # Black
BIRCH_COLOR_HEX = '#31AC73'
UNDERSTORY_COLOR_HEX = '#cdcdcd'
NON_GREEN_COLOR_HEX = '#000000'

#%%
# Create output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

#%% Helper functions
def get_image_rgb_datetime(image_path):
    """Extract datetime from RGB image EXIF"""
    try:
        with Image.open(image_path) as img:
            exifdata = img._getexif()
            if exifdata is None:
                return None
            datetime_taken = None
            for tag_id, value in exifdata.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'DateTimeOriginal' or tag == 'DateTime':
                    datetime_taken = value
                    break
            if datetime_taken:
                try:
                    dt = datetime.datetime.strptime(datetime_taken, '%Y:%m:%d %H:%M:%S')
                    return dt.strftime('%Y-%m-%d %H:%M:%S')
                except ValueError:
                    return datetime_taken
            return None
    except Exception as e:
        print(f"Error reading RGB datetime: {e}")
        return None

def get_image_thermal_datetime(image_path):
    """Extract datetime from thermal .mat filename"""
    try:
        base_name = os.path.basename(image_path)
        name, _ = os.path.splitext(base_name)
        parts = name.split('_')
        try:
            date_part = parts[-2]
            time_part = parts[-1]
            time_part = time_part.replace('.', ':')
            datetime_str = f"{date_part} {time_part}"
            dt = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            return None
    except Exception as e:
        print(f"Error reading thermal datetime: {e}")
        return None

def get_image_position(image_path, image_type):
    """Extract position from image path/filename"""
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

def mask_and_crop_image(img):
    """Crop image to ROI"""
    crop_top = 800
    buffer = 50
    if img.ndim == 3:
        img = img[crop_top:, :, :]
        img = img[buffer:-buffer, buffer:-buffer, :]
    else:
        img = img[crop_top:, :]
        img = img[buffer:-buffer, buffer:-buffer]
    return img

def load_supervised_model(model_path):
    """Load the trained Random Forest classifier"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        if isinstance(model_data, dict):
            classifier = model_data['classifier']
            print(f"Loaded Random Forest model with {model_data.get('n_samples', 'unknown')} training samples")
            print(f"Training accuracy: {model_data.get('training_accuracy', 'unknown'):.4f}")
            print(f"Test accuracy: {model_data.get('test_accuracy', 'unknown'):.4f}")
        else:
            classifier = model_data
            print("Loaded Random Forest model (legacy format)")
        
        return classifier
    except Exception as e:
        print(f"Error loading Random Forest model: {e}")
        return None

def quantify_vegetation_rf_only(img_rgb, img_thermal, classifier):
    """
    Quantifies vegetation using only Random Forest classifier.
    
    Classes:
    - Birch (class 0): Typically brighter, higher L* values in LAB space
    - Understory (class 1): Typically darker vegetation
    - Non-green (class 2): Non-vegetation areas
    """
    try:
        # Convert entire image to LAB color space
        lab_image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB)
        
        # Reshape all pixels for classification
        h, w = img_rgb.shape[:2]
        pixels = lab_image.reshape(-1, 3).astype(np.float32)
        
        # Predict classes using Random Forest for all pixels
        labels = classifier.predict(pixels)
        
        # Reshape back to image dimensions
        label_map = labels.reshape(h, w)
        
        # Create masks for each class
        birch_mask = (label_map == 0)
        understory_mask = (label_map == 1)
        non_green_mask = (label_map == 2)
        green_mask = (birch_mask | understory_mask)
        
        # Calculate greenness for all pixels
        b, g, r = cv2.split(img_rgb)
        b, g, r = b.astype(float), g.astype(float), r.astype(float)
        greenness = g / (r + g + b + 1e-10)
        
        # Calculate pixel counts and ratios
        total_pixels = h * w
        birch_pixels = np.sum(birch_mask)
        understory_pixels = np.sum(understory_mask)
        green_pixels = np.sum(green_mask)
        
        birch_ratio = birch_pixels / total_pixels if total_pixels > 0 else 0
        understory_ratio = understory_pixels / total_pixels if total_pixels > 0 else 0
        green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0
        
        # Compute birch metrics
        if birch_pixels > 0:
            birch_mean = np.mean(greenness[birch_mask])
            birch_std = np.std(greenness[birch_mask])
            birch_mean_temperature = np.nanmean(img_thermal[birch_mask])
            birch_std_temperature = np.nanstd(img_thermal[birch_mask])
        else:
            birch_mean = 0
            birch_std = 0
            birch_mean_temperature = np.nan
            birch_std_temperature = np.nan
        
        birch_metrics = {
            'ratio': birch_ratio,
            'mean': birch_mean,
            'std': birch_std,
            'mean_temperature': birch_mean_temperature,
            'std_temperature': birch_std_temperature
        }
        
        # Compute understory metrics
        if understory_pixels > 0:
            understory_mean = np.mean(greenness[understory_mask])
            understory_std = np.std(greenness[understory_mask])
            understory_mean_temperature = np.nanmean(img_thermal[understory_mask])
            understory_std_temperature = np.nanstd(img_thermal[understory_mask])
        else:
            understory_mean = 0
            understory_std = 0
            understory_mean_temperature = np.nan
            understory_std_temperature = np.nan
        
        understory_metrics = {
            'ratio': understory_ratio,
            'mean': understory_mean,
            'std': understory_std,
            'mean_temperature': understory_mean_temperature,
            'std_temperature': understory_std_temperature
        }
        
        # Overall green metrics
        green_metrics = {
            'ratio': green_ratio,
            'mean': np.mean(greenness[green_mask]) if green_pixels > 0 else 0,
            'std': np.std(greenness[green_mask]) if green_pixels > 0 else 0,
        }
        
        # Create visualization with specified colors
        visualization = np.zeros_like(img_rgb)
        visualization[birch_mask] = BIRCH_COLOR
        visualization[understory_mask] = UNDERSTORY_COLOR
        visualization[non_green_mask] = NON_GREEN_COLOR
        
        # Create masks (uint8)
        green_mask_uint8 = (green_mask.astype(np.uint8) * 255)
        birch_mask_uint8 = (birch_mask.astype(np.uint8) * 255)
        understory_mask_uint8 = (understory_mask.astype(np.uint8) * 255)
        
        return (green_metrics, green_mask_uint8, birch_metrics, understory_metrics, 
                visualization, birch_mask_uint8, understory_mask_uint8)

    except Exception as e:
        print(f"Error in RF-only classification: {e}")
        return None, None, None, None, None, None, None

#%% Main processing
if __name__ == '__main__':
    print("="*70)
    print("Processing Single RGB Image with Median of Nearest Thermal Images")
    print("="*70)
    
    # Load Random Forest classifier
    print(f"\nLoading Random Forest model from {SUPERVISED_MODEL_PATH}...")
    classifier = load_supervised_model(SUPERVISED_MODEL_PATH)
    if classifier is None:
        print("ERROR: Could not load Random Forest model. Exiting.")
        exit(1)
    
    # Load RGB image
    print(f"\nLoading RGB image: {RGB_IMAGE_PATH}")
    img_rgb = cv2.imread(RGB_IMAGE_PATH)
    if img_rgb is None:
        print("ERROR: Could not load RGB image. Exiting.")
        exit(1)
    
    # Get RGB datetime and position
    img_rgb_datetime = get_image_rgb_datetime(RGB_IMAGE_PATH)
    img_rgb_position = get_image_position(RGB_IMAGE_PATH, image_type='rgb')
    print(f"RGB datetime: {img_rgb_datetime}")
    print(f"RGB position: {img_rgb_position}")
    
    # Crop RGB
    img_rgb = mask_and_crop_image(img=img_rgb)
    
    # Load all thermal images
    print(f"\nSearching for thermal images in {THERMAL_FOLDER}...")
    imthermalfiles = glob.glob(os.path.join(THERMAL_FOLDER, '**/*.mat'), recursive=True)
    print(f"Found {len(imthermalfiles)} thermal .mat files")
    
    # Prepare thermal dataframe
    print("Processing thermal file metadata...")
    imthermalfiles_df = pd.DataFrame({'file': imthermalfiles})
    imthermalfiles_df['datetime'] = imthermalfiles_df['file'].apply(get_image_thermal_datetime)
    imthermalfiles_df['position'] = imthermalfiles_df['file'].apply(lambda x: get_image_position(x, image_type='thermal'))
    
    # Filter thermal images by position and valid datetime
    imthermalfiles_filtered = imthermalfiles_df[imthermalfiles_df['position'] == img_rgb_position].copy()
    imthermalfiles_filtered = imthermalfiles_filtered.dropna(subset=['datetime'])
    
    if img_rgb_datetime is None:
        print("WARNING: Could not extract RGB datetime. Cannot find nearest thermal images.")
        exit(1)
    
    if imthermalfiles_filtered.empty:
        print(f"ERROR: No thermal images found for position {img_rgb_position}. Exiting.")
        exit(1)
    
    # Find nearest thermal images
    print(f"\nFinding {NUM_NEAREST_THERMAL} nearest thermal images...")
    img_rgb_datetime_dt = pd.to_datetime(img_rgb_datetime)
    imthermalfiles_filtered['time_diff'] = imthermalfiles_filtered['datetime'].apply(
        lambda x: abs((pd.to_datetime(x) - img_rgb_datetime_dt).total_seconds())
    )
    nearest_thermal = imthermalfiles_filtered.nsmallest(NUM_NEAREST_THERMAL, 'time_diff')
    
    print(f"Found {len(nearest_thermal)} nearest thermal images:")
    for idx, row in nearest_thermal.iterrows():
        print(f"  - {os.path.basename(row['file'])} (time diff: {row['time_diff']:.0f}s)")
    
    # Build vegetation masks from RGB (using dummy thermal)
    print("\nBuilding vegetation masks from RGB...")
    dummy_thermal = np.full(img_rgb.shape[:2], np.nan, dtype=float)
    (green_metrics, green_mask, birch_metrics, understory_metrics, 
     class_vis, birch_mask, understory_mask) = quantify_vegetation_rf_only(
        img_rgb, dummy_thermal, classifier
    )
    
    if green_metrics is None:
        print("ERROR: Classification failed. Exiting.")
        exit(1)
    
    print(f"\nInitial classification (without thermal):")
    print(f"  Green ratio: {green_metrics['ratio']:.4f}")
    print(f"  Birch ratio: {birch_metrics['ratio']:.4f}")
    print(f"  Understory ratio: {understory_metrics['ratio']:.4f}")
    
    # Load and compute median thermal image
    print(f"\nLoading and computing median thermal image...")
    thermal_images = []
    for row in nearest_thermal.itertuples(index=False):
        thermal_file = row.file
        
        # Load thermal
        thermal_mat = loadmat(thermal_file)
        if 'thermal_image_registered' in thermal_mat:
            img_thermal = thermal_mat['thermal_image_registered']
        elif 'thermal_image' in thermal_mat:
            img_thermal = thermal_mat['thermal_image']
        else:
            print(f"  WARNING: Could not find thermal image in {os.path.basename(thermal_file)}. Skipping.")
            continue
        
        # Process thermal
        img_thermal = mask_and_crop_image(img=img_thermal)
        img_thermal = np.double(img_thermal) / 100.0
        img_thermal[img_thermal == 0] = np.nan
        img_thermal = img_thermal - 273.15
        
        thermal_images.append(img_thermal)
    
    if not thermal_images:
        print("ERROR: Could not load any thermal images. Exiting.")
        exit(1)
    
    # Compute median thermal image
    thermal_stack = np.stack(thermal_images, axis=2)
    img_thermal_median = np.nanmedian(thermal_stack, axis=2)
    
    print(f"Computed median from {len(thermal_images)} thermal images")
    
    # Compute thermal statistics using pre-computed masks and median thermal
    bm = birch_mask > 0
    um = understory_mask > 0
    
    birch_temp_mean = np.nanmean(img_thermal_median[bm]) if np.any(bm) else np.nan
    birch_temp_std = np.nanstd(img_thermal_median[bm]) if np.any(bm) else np.nan
    understory_temp_mean = np.nanmean(img_thermal_median[um]) if np.any(um) else np.nan
    understory_temp_std = np.nanstd(img_thermal_median[um]) if np.any(um) else np.nan
    
    print(f"\nMedian thermal statistics:")
    print(f"  Birch temperature: {birch_temp_mean:.2f} ± {birch_temp_std:.2f} °C")
    print(f"  Understory temperature: {understory_temp_mean:.2f} ± {understory_temp_std:.2f} °C")
    
    # Create single visualization
    print("\nCreating visualization...")
    masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=green_mask // 255)
    img_rgb_vis = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    masked_img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
    class_vis_rgb = cv2.cvtColor(class_vis, cv2.COLOR_BGR2RGB)
    
    # Use GridSpec with an extra narrow column reserved for the colorbar so
    # the three main axes (RGB, classification, thermal) keep equal widths.
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.06])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 3])  # colorbar axis

    # Original RGB
    ax0.imshow(img_rgb_vis)
    ax0.set_title(f"a) RGB Image: {img_rgb_datetime or 'NA'}")
    ax0.axis('off')

    # Classification result
    ax1.imshow(class_vis_rgb)
    # ax1.set_title(
    #     f"b) Birch: {birch_metrics['ratio']:.2f} (T={birch_temp_mean:.1f}±{birch_temp_std:.1f}°C)\n"
    #     f"Understory: {understory_metrics['ratio']:.2f} (T={understory_temp_mean:.1f}±{understory_temp_std:.1f}°C)"
    # )
    ax1.set_title(
        f"b) Birch: {birch_temp_mean:.1f}±{birch_temp_std:.1f}°C\n"
        f"Understory: {understory_temp_mean:.1f}±{understory_temp_std:.1f}°C"
    )
    ax1.axis('off')

    # Thermal image (median)
    im = ax2.imshow(img_thermal_median, cmap=cmocean.cm.thermal, vmin=20, vmax=50)
    # Compute average datetime of the (nearest) thermal images for the title
    try:
        times = pd.to_datetime(nearest_thermal['datetime'].dropna())
        mean_time = times.mean() if not times.empty else None
        mean_time_str = mean_time.strftime('%Y-%m-%d %H:%M:%S') if mean_time is not None else 'NA'
    except Exception:
        mean_time_str = 'NA'
    ax2.set_title(f"c) Thermal Image: {mean_time_str}")
    ax2.axis('off')

    # Colorbar using the reserved axis so it does not shrink the thermal axis
    cbar = fig.colorbar(im, cax=cax, shrink=0.6)
    cbar.set_label('Median Temperature (°C)', rotation=270, labelpad=15)
    # cbar.set_ticks(np.linspace(20, 50, 7))

    # Legend centered below the subplots
    legend_elements = [
        Patch(facecolor=BIRCH_COLOR_HEX, edgecolor='black', label='Birch'),
        Patch(facecolor=UNDERSTORY_COLOR_HEX, edgecolor='black', label='Understory'),
        Patch(facecolor=NON_GREEN_COLOR_HEX, edgecolor='black', label='Non-green')
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.46, 0.08), ncol=3, frameon=True,
               facecolor='white', edgecolor='black')

    # Leave room for legend
    plt.tight_layout()
    
    base_rgb = img_rgb_datetime.replace(':', '-').replace(' ', '_') if img_rgb_datetime else "unknown"
    output_plot_path = os.path.join(OUTPUT_DIR, f"{base_rgb}_median_thermal.png")
    fig.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {os.path.basename(output_plot_path)}")
    
    print("\n" + "="*70)
    print("Processing complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)