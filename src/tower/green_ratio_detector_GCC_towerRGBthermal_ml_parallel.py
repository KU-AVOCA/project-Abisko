"""
Green Ratio Detector for Vegetation Analysis with Machine Learning Classification (Parallel Version)

This is a parallelized version of the original script for faster processing.
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
warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

sns.set_theme(style="darkgrid", font_scale=1.5)

#%%
# Set the classification method to use
classification_method = "kmeans"  # Options: "kmeans", "gmm", "dbscan", "spectral"

# Set number of parallel processes (None = use all CPUs, or specify a number)
NUM_PROCESSES = 15#cpu_count() - 170  # Leave 170 CPUs free

#%% Global paths (need to be global for multiprocessing)
rgbfolder = '/data/shunan/data/KU/rgbimages/'
thermalfolder = '/data_3/shunan_2/KU/registeredMatImages/'
imoutfolder = '/data_3/shunan_2/KU/Data_greenes_thermal_' + classification_method + '_mean'

#%% Create output directories
if not os.path.exists(imoutfolder):
    os.makedirs(imoutfolder)

results_dir = os.path.join(imoutfolder, 'results')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

masks_dir = os.path.join(results_dir, 'classification_masks')
if not os.path.exists(masks_dir):
    os.makedirs(masks_dir)

csv_path = os.path.join(results_dir, 'green_ratio_thermal_' + classification_method + '.csv')

#%% Helper functions (keep all your existing helper functions here)
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

# ...existing code... (include all your quantify_vegetation functions here)
# I'll include the kmeans one as example, add the others similarly

def quantify_vegetation_kmeans(img_rgb, img_thermal):
    """Quantifies green vegetation using K-means clustering."""
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
        
        masked_green = cv2.bitwise_and(img_rgb, img_rgb, mask=(green_mask // 255).astype(np.uint8))
        non_zero_mask = np.any(masked_green != 0, axis=2)
        
        class1_metrics = {'ratio': 0, 'mean': 0, 'std': 0, 'norm_greenness': 0}
        class2_metrics = {'ratio': 0, 'mean': 0, 'std': 0, 'norm_greenness': 0}
        visualization = np.zeros_like(img_rgb)
        
        if np.sum(non_zero_mask) > 0:
            lab_image = cv2.cvtColor(masked_green, cv2.COLOR_BGR2LAB)
            pixels = lab_image[non_zero_mask].reshape(-1, 3).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            full_labels = np.full(img_rgb.shape[:2], -1, dtype=int)
            full_labels[non_zero_mask] = labels.flatten()
            
            if centers[0][0] < centers[1][0]:
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
        return None, None, None, None, None

# Add your other quantify_vegetation_* functions here (gmm, dbscan, spectral)

def quantify_vegetation(img_rgb, img_thermal, method="kmeans"):
    """Wrapper function for vegetation quantification"""
    if method == "kmeans":
        return quantify_vegetation_kmeans(img_rgb, img_thermal)
    # Add other methods here
    else:
        return quantify_vegetation_kmeans(img_rgb, img_thermal)

def func_compute_thermal_stats_from_masks(img_thermal, green_mask, class1_mask, class2_mask):
    """Compute thermal statistics from masks"""
    gm = green_mask > 0
    green_temp_mean = np.nanmean(img_thermal[gm]) if np.any(gm) else np.nan
    green_temp_std = np.nanstd(img_thermal[gm]) if np.any(gm) else np.nan
    
    c1 = class1_mask.astype(bool)
    class1_temp_mean = np.nanmean(img_thermal[c1]) if np.any(c1) else np.nan
    class1_temp_std = np.nanstd(img_thermal[c1]) if np.any(c1) else np.nan
    
    c2 = class2_mask.astype(bool)
    class2_temp_mean = np.nanmean(img_thermal[c2]) if np.any(c2) else np.nan
    class2_temp_std = np.nanstd(img_thermal[c2]) if np.any(c2) else np.nan
    
    return (green_temp_mean, green_temp_std,
            class1_temp_mean, class1_temp_std,
            class2_temp_mean, class2_temp_std)

#%% Main processing function for parallel execution
def process_single_rgb_image(args):
    """
    Process a single RGB image with its matching thermal images.
    This function is called in parallel for each RGB image.
    """
    rgb_file, imthermalfiles_df, classification_method = args
    
    results = []  # Store results for this RGB image
    
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
        green_metrics, green_mask, class1_metrics, class2_metrics, class_vis = quantify_vegetation(
            img_rgb, dummy_thermal, method=classification_method
        )
        
        if green_metrics is None:
            return []
        
        # Extract class masks
        class1_mask = (class_vis[:, :, 0] == 255) & (class_vis[:, :, 1] == 0) & (class_vis[:, :, 2] == 0)
        class2_mask = (class_vis[:, :, 0] == 0) & (class_vis[:, :, 1] == 255) & (class_vis[:, :, 2] == 0)
        
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
             class1_temp_mean, class1_temp_std,
             class2_temp_mean, class2_temp_std) = func_compute_thermal_stats_from_masks(
                img_thermal, green_mask, class1_mask, class2_mask
            )
            
            # Update metrics
            gm = dict(green_metrics)
            gm['mean_temperature'] = green_temp_mean
            gm['std_temperature'] = green_temp_std
            
            c1m = dict(class1_metrics)
            c1m['mean_temperature'] = class1_temp_mean
            c1m['std_temperature'] = class1_temp_std
            
            c2m = dict(class2_metrics)
            c2m['mean_temperature'] = class2_temp_mean
            c2m['std_temperature'] = class2_temp_std
            
            # Save masks
            base_rgb = datetime_str.replace(':', '-').replace(' ', '_') if datetime_str != "NA" else os.path.splitext(os.path.basename(rgb_file))[0]
            base_therm = therm_dt_str.replace(':', '-').replace(' ', '_') if therm_dt_str != "NA" else os.path.splitext(os.path.basename(thermal_file))[0]
            mask_filename = f"{base_rgb}__therm_{base_therm}_{classification_method}_masks.mat"
            mask_filepath = os.path.join(masks_dir, mask_filename)
            
            savemat(mask_filepath, {
                'green_mask': green_mask.astype(np.uint8),
                'class1_mask': class1_mask.astype(np.uint8),
                'class2_mask': class2_mask.astype(np.uint8),
                'metadata': {
                    'datetime_rgb': datetime_str,
                    'datetime_thermal': therm_dt_str,
                    'method': classification_method,
                    'rgb_file': rgb_file,
                    'thermal_file': thermal_file,
                    'green_ratio': gm['ratio'],
                    'class1_ratio': c1m['ratio'],
                    'class2_ratio': c2m['ratio'],
                    'class1_temp_mean': c1m['mean_temperature'],
                    'class2_temp_mean': c2m['mean_temperature']
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
                'class1_ratio': c1m["ratio"],
                'class1_mean': c1m["mean"],
                'class1_std': c1m["std"],
                'class1_norm': c1m["norm_greenness"],
                'class2_ratio': c2m["ratio"],
                'class2_mean': c2m["mean"],
                'class2_std': c2m["std"],
                'class2_norm': c2m["norm_greenness"],
                'class1_temp_mean': c1m["mean_temperature"],
                'class1_temp_std': c1m["std_temperature"],
                'class2_temp_mean': c2m["mean_temperature"],
                'class2_temp_std': c2m["std_temperature"],
                'time_diff_sec': time_diff_seconds,
                'method': classification_method,
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
                f"Vegetation Classes ({classification_method})\n"
                f"class1={c1m['ratio']:.2f} (Temp={c1m['mean_temperature']:.2f}±{c1m['std_temperature']:.2f})\n"
                f"class2={c2m['ratio']:.2f} (Temp={c2m['mean_temperature']:.2f}±{c2m['std_temperature']:.2f})"
            )
            axes[2].axis('off')
            
            legend_elements = [
                Patch(facecolor='blue', edgecolor='black', label='Class 1'),
                Patch(facecolor='green', edgecolor='black', label='Class 2')
            ]
            axes[2].legend(handles=legend_elements, loc='lower center',
                           bbox_to_anchor=(0.5, -0.4), frameon=True,
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
    
    # Filter to West-facing only (temporary)
    imrgbfiles = [f for f in imrgbfiles if 'West' in f]
    print(f"Processing {len(imrgbfiles)} West-facing RGB images")
    
    # Initialize CSV file
    with open(csv_path, 'w') as f:
        f.write('filename,datetime,green_ratio,green_mean,green_std,green_norm,class1_ratio,class1_mean,class1_std,class1_norm,class2_ratio,class2_mean,class2_std,class2_norm,class1_temp_mean,class1_temp_std,class2_temp_mean,class2_temp_std,time_diff_sec,method,mask_file\n')
    
    # Prepare arguments for parallel processing
    args_list = [(rgb_file, imthermalfiles_df, classification_method) for rgb_file in imrgbfiles]
    
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
                               f'{result["class1_ratio"]},{result["class1_mean"]},{result["class1_std"]},{result["class1_norm"]},'
                               f'{result["class2_ratio"]},{result["class2_mean"]},{result["class2_std"]},{result["class2_norm"]},'
                               f'{result["class1_temp_mean"]},{result["class1_temp_std"]},'
                               f'{result["class2_temp_mean"]},{result["class2_temp_std"]},'
                               f'{result["time_diff_sec"]},{result["method"]},{result["mask_file"]}\n')
    
    print(f"\nDone! Processed {len(all_results)} RGB-thermal pairs")
    print(f"Results saved to {csv_path}")
    print(f"Classification masks saved to {masks_dir}")