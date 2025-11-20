"""
Green Ratio Detector for Vegetation Analysis with Random Forest Classification (Parallel Version)

This script uses a trained Random Forest classifier to distinguish birch and understory vegetation.
Uses multiprocessing to process multiple RGB images simultaneously.

Author: Shunan Feng (shf@ign.ku.dk)
"""
#%%
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for parallel processing
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
from multiprocessing import Pool, cpu_count
import warnings
import pickle
warnings.filterwarnings('ignore')

sns.set_theme(style="darkgrid", font_scale=1.5)

#%%
# Path to trained Random Forest model
SUPERVISED_MODEL_PATH = "src/tower/vegetation_classifier.pkl"

# Set number of parallel processes
NUM_PROCESSES = 20

# Color scheme for visualization (RGB format)
BIRCH_COLOR = (115, 172, 49)  # #73ac31 in RGB
UNDERSTORY_COLOR = (205, 205, 205)  # #cdcdcd in RGB
NON_GREEN_COLOR = (0, 0, 0)  # Black
BIRCH_COLOR_HEX = '#73ac31'
UNDERSTORY_COLOR_HEX = '#cdcdcd'
NON_GREEN_COLOR_HEX = '#000000'

#%% Global paths
rgbfolder = '/data/shunan/data/KU/rgbimages/'
thermalfolder = '/data_3/shunan_2/KU/registeredMatImages/'
imoutfolder = '/data_3/shunan_2/KU/Data_greennes_thermal_RF_supervised' 

#%% Create output directories
if not os.path.exists(imoutfolder):
    os.makedirs(imoutfolder)

results_dir = os.path.join(imoutfolder, 'results')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

masks_dir = os.path.join(results_dir, 'classification_masks')
if not os.path.exists(masks_dir):
    os.makedirs(masks_dir)

csv_path = os.path.join(results_dir, 'green_ratio_thermal_RF_supervised.csv')

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

def quantify_vegetation_supervised(img_rgb, img_thermal, classifier):
    """
    Quantifies vegetation using supervised Random Forest classifier.
    
    Classes:
    - Birch (class 0): Typically brighter, higher L* values in LAB space
    - Understory (class 1): Typically darker vegetation
    - Non-green (class 2): Not used in final classification (filtered by green mask)
    """
    try:
        # Calculate greenness for initial green mask
        b, g, r = cv2.split(img_rgb)
        b, g, r = b.astype(float), g.astype(float), r.astype(float)
        greenness = g / (r + g + b + 1e-10)
        threshold = 0.38
        green_mask = (greenness > threshold).astype(np.uint8) * 255
        
        green_pixels = np.sum(green_mask > 0)
        total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
        green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0
        
        # Compute overall green metrics
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
        
        # Apply green mask to RGB
        masked_green = cv2.bitwise_and(img_rgb, img_rgb, mask=(green_mask // 255).astype(np.uint8))
        non_zero_mask = np.any(masked_green != 0, axis=2)
        
        # Initialize birch and understory metrics
        birch_metrics = {'ratio': 0, 'mean': 0, 'std': 0, 'norm_greenness': 0, 
                        'mean_temperature': np.nan, 'std_temperature': np.nan}
        understory_metrics = {'ratio': 0, 'mean': 0, 'std': 0, 'norm_greenness': 0,
                             'mean_temperature': np.nan, 'std_temperature': np.nan}
        
        # Initialize visualization with black background (non-green)
        visualization = np.zeros_like(img_rgb)
        visualization[:, :] = NON_GREEN_COLOR
        
        if np.sum(non_zero_mask) > 0 and classifier is not None:
            # Convert to LAB color space for classification
            lab_image = cv2.cvtColor(masked_green, cv2.COLOR_BGR2LAB)
            pixels = lab_image[non_zero_mask].reshape(-1, 3).astype(np.float32)
            
            # Predict classes using Random Forest
            labels = classifier.predict(pixels)
            
            # Create full label map
            full_labels = np.full(img_rgb.shape[:2], -1, dtype=int)
            full_labels[non_zero_mask] = labels
            
            # Extract birch (class 0) and understory (class 1) masks
            birch_mask = (full_labels == 0)
            understory_mask = (full_labels == 1)
            
            # Calculate pixel counts and ratios
            birch_pixels = np.sum(birch_mask)
            understory_pixels = np.sum(understory_mask)
            birch_ratio = birch_pixels / total_pixels if total_pixels > 0 else 0
            understory_ratio = understory_pixels / total_pixels if total_pixels > 0 else 0
            
            # Compute birch metrics
            if birch_pixels > 0:
                birch_mean = np.mean(greenness[birch_mask])
                birch_std = np.std(greenness[birch_mask])
                birch_norm_greenness = np.sum(greenness[birch_mask]) / birch_pixels
                birch_mean_temperature = np.nanmean(img_thermal[birch_mask])
                birch_std_temperature = np.nanstd(img_thermal[birch_mask])
                
                birch_metrics = {
                    'ratio': birch_ratio,
                    'mean': birch_mean,
                    'std': birch_std,
                    'norm_greenness': birch_norm_greenness,
                    'mean_temperature': birch_mean_temperature,
                    'std_temperature': birch_std_temperature
                }
            
            # Compute understory metrics
            if understory_pixels > 0:
                understory_mean = np.mean(greenness[understory_mask])
                understory_std = np.std(greenness[understory_mask])
                understory_norm_greenness = np.sum(greenness[understory_mask]) / understory_pixels
                understory_mean_temperature = np.nanmean(img_thermal[understory_mask])
                understory_std_temperature = np.nanstd(img_thermal[understory_mask])
                
                understory_metrics = {
                    'ratio': understory_ratio,
                    'mean': understory_mean,
                    'std': understory_std,
                    'norm_greenness': understory_norm_greenness,
                    'mean_temperature': understory_mean_temperature,
                    'std_temperature': understory_std_temperature
                }
            
            # Create visualization with specified colors
            # Birch = #73ac31, Understory = #cdcdcd, Non-green = Black
            visualization[birch_mask] = BIRCH_COLOR
            visualization[understory_mask] = UNDERSTORY_COLOR
        
        return green_metrics, green_mask, birch_metrics, understory_metrics, visualization

    except Exception as e:
        print(f"Error in supervised classification: {e}")
        return None, None, None, None, None

def func_compute_thermal_stats_from_masks(img_thermal, green_mask, birch_mask, understory_mask):
    """Compute thermal statistics from masks"""
    gm = green_mask > 0
    green_temp_mean = np.nanmean(img_thermal[gm]) if np.any(gm) else np.nan
    green_temp_std = np.nanstd(img_thermal[gm]) if np.any(gm) else np.nan
    
    bm = birch_mask.astype(bool)
    birch_temp_mean = np.nanmean(img_thermal[bm]) if np.any(bm) else np.nan
    birch_temp_std = np.nanstd(img_thermal[bm]) if np.any(bm) else np.nan
    
    um = understory_mask.astype(bool)
    understory_temp_mean = np.nanmean(img_thermal[um]) if np.any(um) else np.nan
    understory_temp_std = np.nanstd(img_thermal[um]) if np.any(um) else np.nan
    
    return (green_temp_mean, green_temp_std,
            birch_temp_mean, birch_temp_std,
            understory_temp_mean, understory_temp_std)

#%% Main processing function for parallel execution
def process_single_rgb_image(args):
    """
    Process a single RGB image with its matching thermal images.
    This function is called in parallel for each RGB image.
    """
    rgb_file, imthermalfiles_df, classifier = args
    
    results = []
    
    try:
        # Read RGB image
        img_rgb = cv2.imread(rgb_file)
        if img_rgb is None:
            return []
        
        # Get RGB datetime and position
        img_rgb_datetime = get_image_rgb_datetime(rgb_file)
        datetime_str = img_rgb_datetime if img_rgb_datetime else "NA"
        img_rgb_position = get_image_position(rgb_file, image_type='rgb')
        
        # Crop RGB
        img_rgb = mask_and_crop_image(img=img_rgb)
        
        # Build vegetation masks once from RGB
        dummy_thermal = np.full(img_rgb.shape[:2], np.nan, dtype=float)
        green_metrics, green_mask, birch_metrics, understory_metrics, class_vis = quantify_vegetation_supervised(
            img_rgb, dummy_thermal, classifier
        )
        
        if green_metrics is None:
            return []
        
        # Extract birch and understory masks
        # Birch is #73ac31 (115, 172, 49), Understory is #cdcdcd (205, 205, 205)
        birch_mask = (class_vis[:, :, 0] == BIRCH_COLOR[0]) & (class_vis[:, :, 1] == BIRCH_COLOR[1]) & (class_vis[:, :, 2] == BIRCH_COLOR[2])
        understory_mask = (class_vis[:, :, 0] == UNDERSTORY_COLOR[0]) & (class_vis[:, :, 1] == UNDERSTORY_COLOR[1]) & (class_vis[:, :, 2] == UNDERSTORY_COLOR[2])
        
        # Filter thermal images
        imthermalfiles_filtered = imthermalfiles_df[imthermalfiles_df['position'] == img_rgb_position].copy()
        imthermalfiles_filtered = imthermalfiles_filtered.dropna(subset=['datetime'])
        
        if img_rgb_datetime is None or imthermalfiles_filtered.empty:
            return []
        
        # Find 5 closest thermal images
        img_rgb_datetime_dt = pd.to_datetime(img_rgb_datetime)
        imthermalfiles_filtered['time_diff'] = imthermalfiles_filtered['datetime'].apply(
            lambda x: abs((pd.to_datetime(x) - img_rgb_datetime_dt).total_seconds())
        )
        nearest5 = imthermalfiles_filtered.nsmallest(5, 'time_diff')
        
        if nearest5.empty:
            return []
        
        # Precompute visualization images
        masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=green_mask // 255)
        img_rgb_vis = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        masked_img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
        class_vis_rgb = cv2.cvtColor(class_vis, cv2.COLOR_BGR2RGB)
        
        # Process each thermal image
        for row in nearest5.itertuples(index=False):
            thermal_file = row.file
            time_diff_seconds = row.time_diff
            therm_dt_str = get_image_thermal_datetime(thermal_file) or "NA"
            
            # Load thermal
            thermal_mat = loadmat(thermal_file)
            if 'thermal_image_registered' in thermal_mat:
                img_thermal = thermal_mat['thermal_image_registered']
            elif 'thermal_image' in thermal_mat:
                img_thermal = thermal_mat['thermal_image']
            else:
                continue
            
            # Process thermal
            img_thermal = mask_and_crop_image(img=img_thermal)
            img_thermal = np.double(img_thermal) / 100.0
            img_thermal[img_thermal == 0] = np.nan
            img_thermal = img_thermal - 273.15
            
            # Compute thermal stats
            (green_temp_mean, green_temp_std,
             birch_temp_mean, birch_temp_std,
             understory_temp_mean, understory_temp_std) = func_compute_thermal_stats_from_masks(
                img_thermal, green_mask, birch_mask, understory_mask
            )
            
            # Update metrics with thermal data
            gm = dict(green_metrics)
            gm['mean_temperature'] = green_temp_mean
            gm['std_temperature'] = green_temp_std
            
            bm = dict(birch_metrics)
            bm['mean_temperature'] = birch_temp_mean
            bm['std_temperature'] = birch_temp_std
            
            um = dict(understory_metrics)
            um['mean_temperature'] = understory_temp_mean
            um['std_temperature'] = understory_temp_std
            
            # Save masks
            base_rgb = datetime_str.replace(':', '-').replace(' ', '_') if datetime_str != "NA" else os.path.splitext(os.path.basename(rgb_file))[0]
            base_therm = therm_dt_str.replace(':', '-').replace(' ', '_') if therm_dt_str != "NA" else os.path.splitext(os.path.basename(thermal_file))[0]
            mask_filename = f"{base_rgb}__therm_{base_therm}_RF_masks.mat"
            mask_filepath = os.path.join(masks_dir, mask_filename)
            
            savemat(mask_filepath, {
                'green_mask': green_mask.astype(np.uint8),
                'birch_mask': birch_mask.astype(np.uint8),
                'understory_mask': understory_mask.astype(np.uint8),
                'metadata': {
                    'datetime_rgb': datetime_str,
                    'datetime_thermal': therm_dt_str,
                    'method': 'Random Forest',
                    'rgb_file': rgb_file,
                    'thermal_file': thermal_file,
                    'green_ratio': gm['ratio'],
                    'birch_ratio': bm['ratio'],
                    'understory_ratio': um['ratio'],
                    'birch_temp_mean': bm['mean_temperature'],
                    'understory_temp_mean': um['mean_temperature']
                }
            })
            
            # Create result entry
            result = {
                'filename': rgb_file,
                'datetime': datetime_str,
                'green_ratio': gm["ratio"],
                'green_mean': gm["mean"],
                'green_std': gm["std"],
                'green_norm': gm["norm_greenness"],
                'birch_ratio': bm["ratio"],
                'birch_mean': bm["mean"],
                'birch_std': bm["std"],
                'birch_norm': bm["norm_greenness"],
                'understory_ratio': um["ratio"],
                'understory_mean': um["mean"],
                'understory_std': um["std"],
                'understory_norm': um["norm_greenness"],
                'birch_temp_mean': bm["mean_temperature"],
                'birch_temp_std': bm["std_temperature"],
                'understory_temp_mean': um["mean_temperature"],
                'understory_temp_std': um["std_temperature"],
                'time_diff_sec': time_diff_seconds,
                'method': 'Random Forest',
                'mask_file': mask_filename
            }
            results.append(result)
            
            # Save visualization
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(img_rgb_vis)
            axes[0].set_title(f"Original ({datetime_str})" if datetime_str != "NA" else "Original")
            axes[0].axis('off')
            
            axes[1].imshow(masked_img_rgb)
            axes[1].set_title(f"Green Masked (GR={gm['ratio']:.2f}, Norm={gm['norm_greenness']:.2f})")
            axes[1].axis('off')
            
            axes[2].imshow(class_vis_rgb)
            axes[2].set_title(
                f"Vegetation Classes (Random Forest)\n"
                f"Birch={bm['ratio']:.2f} (Temp={bm['mean_temperature']:.2f}±{bm['std_temperature']:.2f}°C)\n"
                f"Understory={um['ratio']:.2f} (Temp={um['mean_temperature']:.2f}±{um['std_temperature']:.2f}°C)"
            )
            axes[2].axis('off')
            
            # Convert RGB to normalized RGB for matplotlib patches
            # birch_color_norm = tuple(c / 255.0 for c in BIRCH_COLOR)
            # understory_color_norm = tuple(c / 255.0 for c in UNDERSTORY_COLOR)
            # non_green_color_norm = tuple(c / 255.0 for c in NON_GREEN_COLOR)
            
            legend_elements = [
                Patch(facecolor=BIRCH_COLOR_HEX, edgecolor='black', label='Birch'),
                Patch(facecolor=UNDERSTORY_COLOR_HEX, edgecolor='black', label='Understory'),
                Patch(facecolor=NON_GREEN_COLOR_HEX, edgecolor='black', label='Non-green')
            ]
            axes[2].legend(handles=legend_elements, loc='lower center',
                           bbox_to_anchor=(0.5, -0.5), frameon=True,
                           facecolor='white', edgecolor='black')
            
            im = axes[3].imshow(img_thermal, cmap=cmocean.cm.thermal)
            axes[3].set_title('Thermal Image')
            axes[3].axis('off')
            cbar = plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
            cbar.set_label('Temperature (°C)', rotation=270, labelpad=15)
            plt.tight_layout()
            
            rel_path = os.path.relpath(os.path.dirname(rgb_file), rgbfolder)
            output_dir = os.path.join(imoutfolder, rel_path)
            os.makedirs(output_dir, exist_ok=True)
            base_plot = base_rgb + f"__therm_{base_therm}"
            extension = os.path.splitext(os.path.basename(rgb_file))[1]
            output_path = os.path.join(output_dir, f"{base_plot}_green_masked{extension}")
            fig.savefig(output_path, dpi=100)
            plt.close(fig)
    
    except Exception as e:
        print(f"Error processing {rgb_file}: {e}")
        return []
    
    return results

#%% Main execution
if __name__ == '__main__':
    print(f"Using {NUM_PROCESSES} parallel processes")
    print(f"Classification method: Random Forest (Supervised)")
    print(f"Color scheme: Birch=#73ac31, Understory=#cdcdcd, Non-green=Black")
    
    # Load Random Forest classifier
    print(f"\nLoading Random Forest model from {SUPERVISED_MODEL_PATH}...")
    classifier = load_supervised_model(SUPERVISED_MODEL_PATH)
    if classifier is None:
        print("ERROR: Could not load Random Forest model. Exiting.")
        exit(1)
    
    # Load RGB images
    imrgbfiles = []
    imrgbfiles.extend(glob.glob(os.path.join(rgbfolder, '**/', '*.JPG'), recursive=True))
    imrgbfiles.extend(glob.glob(os.path.join(rgbfolder, '**/', '*.jpg'), recursive=True))
    imrgbfiles.extend(glob.glob(os.path.join(rgbfolder, '**/', '*.JPEG'), recursive=True))
    imrgbfiles.extend(glob.glob(os.path.join(rgbfolder, '**/', '*.jpeg'), recursive=True))
    imrgbfiles.extend(glob.glob(os.path.join(rgbfolder, '**/', '*.png'), recursive=True))
    imrgbfiles.extend(glob.glob(os.path.join(rgbfolder, '**/', '*.PNG'), recursive=True))
    print(f"Found {len(imrgbfiles)} RGB images")
    
    # Load thermal images
    imthermalfiles = []
    imthermalfiles.extend(glob.glob(os.path.join(thermalfolder, '**/', '*.mat'), recursive=True))
    print(f"Found {len(imthermalfiles)} thermal .mat files")
    
    # Prepare thermal dataframe
    imthermalfiles_df = pd.DataFrame({'file': imthermalfiles})
    imthermalfiles_df['datetime'] = imthermalfiles_df['file'].apply(get_image_thermal_datetime)
    imthermalfiles_df['position'] = imthermalfiles_df['file'].apply(lambda x: get_image_position(x, image_type='thermal'))
    
    # Filter to West-facing only
    imrgbfiles = [f for f in imrgbfiles if 'West' in f]
    print(f"Processing {len(imrgbfiles)} West-facing RGB images")
    
    # Initialize CSV file with consistent column names
    with open(csv_path, 'w') as f:
        f.write('filename,datetime,green_ratio,green_mean,green_std,green_norm,'
                'birch_ratio,birch_mean,birch_std,birch_norm,'
                'understory_ratio,understory_mean,understory_std,understory_norm,'
                'birch_temp_mean,birch_temp_std,understory_temp_mean,understory_temp_std,'
                'time_diff_sec,method,mask_file\n')
    
    # Prepare arguments for parallel processing
    args_list = [(rgb_file, imthermalfiles_df, classifier) for rgb_file in imrgbfiles]
    
    # Process in parallel with progress bar
    all_results = []
    with Pool(processes=NUM_PROCESSES) as pool:
        for results in tqdm.tqdm(pool.imap_unordered(process_single_rgb_image, args_list), 
                                 total=len(args_list), 
                                 desc="Processing RGB images"):
            all_results.extend(results)
            
            # Write results immediately to CSV
            if results:
                with open(csv_path, 'a') as f:
                    for result in results:
                        f.write(f'{result["filename"]},{result["datetime"]},'
                               f'{result["green_ratio"]},{result["green_mean"]},{result["green_std"]},{result["green_norm"]},'
                               f'{result["birch_ratio"]},{result["birch_mean"]},{result["birch_std"]},{result["birch_norm"]},'
                               f'{result["understory_ratio"]},{result["understory_mean"]},{result["understory_std"]},{result["understory_norm"]},'
                               f'{result["birch_temp_mean"]},{result["birch_temp_std"]},'
                               f'{result["understory_temp_mean"]},{result["understory_temp_std"]},'
                               f'{result["time_diff_sec"]},{result["method"]},{result["mask_file"]}\n')
    
    print(f"\nDone! Processed {len(all_results)} RGB-thermal pairs")
    print(f"Results saved to {csv_path}")
    print(f"Classification masks saved to {masks_dir}")