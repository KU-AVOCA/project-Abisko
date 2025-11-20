'''
Dynamic Threshold Detector for Tower RGB Images (Parallel Version)

This script processes tower RGB images to determine an optimal GCC threshold
by calculating thresholds for individual images and analyzing their distribution.

Based on green_ratio_detector_GCCdynamicthreshold.py but adapted for tower images
with parallel processing for improved performance.

Shunan Feng (shf@ign.ku.dk)
'''
#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import tqdm
import seaborn as sns
from scipy import stats
import concurrent.futures
from functools import partial
import multiprocessing

# Set matplotlib to non-interactive backend for parallel processing
import matplotlib
matplotlib.use('Agg')

sns.set_theme(style="darkgrid", font_scale=1.5)

#%%
# Input folder for tower RGB images
imfolder = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/1_Simon/1_Abisko/6_Tower_Data/Tower RGB images/1 Data/1 Years'
imfiles = []
imfiles.extend(glob.glob(os.path.join(imfolder, '**/', '*.JPG'), recursive=True))
imfiles.extend(glob.glob(os.path.join(imfolder, '**/', '*.jpg'), recursive=True))
imfiles.extend(glob.glob(os.path.join(imfolder, '**/', '*.JPEG'), recursive=True))
imfiles.extend(glob.glob(os.path.join(imfolder, '**/', '*.jpeg'), recursive=True))
imfiles.extend(glob.glob(os.path.join(imfolder, '**/', '*.png'), recursive=True))
imfiles.extend(glob.glob(os.path.join(imfolder, '**/', '*.PNG'), recursive=True))

print(f"Found {len(imfiles)} images in total")

# Output folder
imoutfolder = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/Tower RGB images/threshold_determination/'
if not os.path.exists(imoutfolder):
    os.makedirs(imoutfolder)

#%%    
def quantify_vegetation(img, threshold):
    """
    Quantifies the ratio of green to non-green pixels using Greenness index (GCC).

    Args:
        img: The input image (BGR format)
        threshold: GCC threshold value

    Returns:
        tuple: Ratio of green pixels to total pixels, and the green mask.
    """
    try:
        # Split the image into its BGR channels
        b, g, r = cv2.split(img)
        
        # Convert to float to avoid integer division
        b = b.astype(float)
        g = g.astype(float)
        r = r.astype(float)
        
        # Calculate the Greenness index G/(R+G+B)
        greenness = g / (r + g + b + 1e-10)

        # Create a binary mask using the threshold
        green_mask = (greenness > threshold).astype(np.uint8) * 255
        
        # Count green pixels and calculate ratio
        green_pixels = np.sum(green_mask > 0)
        total_pixels = img.shape[0] * img.shape[1]
        green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0
        
        return green_ratio, green_mask

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def determine_threshold_otsu(img):
    """
    Determine optimal GCC threshold using Otsu's method.
    
    Args:
        img: Input BGR image
        
    Returns:
        float: Optimal threshold value for GCC
    """
    # Calculate GCC
    b, g, r = cv2.split(img)
    b = b.astype(float)
    g = g.astype(float)
    r = r.astype(float)
    greenness = g / (r + g + b + 1e-10)
    
    # Scale to 0-255 for Otsu
    greenness_scaled = (greenness * 255).astype(np.uint8)
    
    # Apply Otsu's method
    threshold_value, _ = cv2.threshold(greenness_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert back to original GCC scale (0-1)
    threshold = threshold_value / 255.0
    
    # Ensure threshold is at least 0.2
    threshold = max(threshold, 0.2)
    
    return threshold


def determine_threshold_kmeans(img, n_clusters=2):
    """
    Use K-means clustering to determine GCC threshold, separating
    vegetation from non-vegetation.
    
    Args:
        img: Input BGR image
        n_clusters: Number of clusters (2 for vegetation/non-vegetation)
        
    Returns:
        float: Determined threshold for GCC
    """
    # Calculate GCC
    b, g, r = cv2.split(img)
    b, g, r = b.astype(float), g.astype(float), r.astype(float)
    greenness = g / (r + g + b + 1e-10)
    
    # Reshape for clustering
    data = greenness.reshape(-1, 1).astype(np.float32)
    
    # Apply K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(data, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Sort centers (low to high)
    sorted_indices = np.argsort(centers.flatten())
    
    # The threshold is between the non-vegetation and vegetation clusters
    lower_center = centers[sorted_indices[0]][0]
    higher_center = centers[sorted_indices[1]][0]
    
    # Use a weighted position between clusters
    threshold = lower_center + 0.6 * (higher_center - lower_center)
    
    return threshold


def remove_outliers_mad(df, column='threshold'):
    """
    Remove outliers using Median Absolute Deviation from scipy.stats - more robust than IQR
    """
    
    median = df[column].median()
    # Calculate MAD using scipy.stats
    mad = stats.median_abs_deviation(df[column], scale=1)

    q75_scale = 1 / df[column].quantile(0.75)
    
    # Scale factor for normal distribution (1.4826 for normal distribution)
    mad_scaled = mad * 1.4826 #q75_scale #
    
    # Define bounds
    lower_bound = median - mad_scaled
    upper_bound = median + mad_scaled
    
    print(f"MAD bounds: {lower_bound:.4f} to {upper_bound:.4f}")
    print(f"Q75: {df[column].quantile(0.75)}")
    print(f"Q25: {df[column].quantile(0.25)}")
    
    # Filter dataframe
    filtered_df = df[(df[column] > lower_bound) & (df[column] < upper_bound)]
    print(f"Removed {len(df) - len(filtered_df)} outliers using MAD method")
    
    return filtered_df


def process_single_image(image_path, output_folder):
    """
    Process a single image and return results.
    
    Args:
        image_path: Path to the image file
        output_folder: Output folder for saving visualizations
        
    Returns:
        dict: Dictionary containing filename, green_ratio, threshold, and status
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {
                'filename': image_path, 
                'green_ratio': None, 
                'threshold': None,
                'status': 'failed',
                'error': 'Could not read image'
            }

        # Determine threshold using Otsu's method
        threshold = determine_threshold_otsu(img)
        # Alternative: threshold = determine_threshold_kmeans(img, n_clusters=2)
        
        ratio, green_mask = quantify_vegetation(img, threshold)

        if ratio is not None:
            # Apply the green mask to the image
            masked_img = cv2.bitwise_and(img, img, mask=green_mask)

            # Convert the images to RGB format for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            masked_img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)

            # Create a figure and axes
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Display the original image
            axes[0].imshow(img_rgb)
            axes[0].set_title("Original Image")
            axes[0].axis('off')

            # Display the masked image
            axes[1].imshow(masked_img_rgb)
            axes[1].set_title("Green Masked Image")
            axes[1].axis('off')

            plt.tight_layout()

            # Save the figure
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            extension = os.path.splitext(os.path.basename(image_path))[1]
            output_path = os.path.join(output_folder, f"{base_filename}_green_masked{extension}")
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)  # Important: close figure to free memory

            return {
                'filename': image_path,
                'green_ratio': ratio,
                'threshold': threshold,
                'status': 'success'
            }
        else:
            return {
                'filename': image_path,
                'green_ratio': None,
                'threshold': None,
                'status': 'failed',
                'error': 'Vegetation quantification failed'
            }

    except Exception as e:
        return {
            'filename': image_path,
            'green_ratio': None,
            'threshold': None,
            'status': 'failed',
            'error': str(e)
        }


def process_images_parallel(image_files, output_folder, max_workers=None):
    """
    Process images in parallel using ProcessPoolExecutor.
    
    Args:
        image_files: List of image file paths
        output_folder: Output folder for saving visualizations
        max_workers: Maximum number of worker processes (default: CPU count - 2)
        
    Returns:
        list: List of dictionaries containing results for each image
    """
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 2)
    
    print(f"Using {max_workers} worker processes")
    
    # Create a partial function with the fixed output_folder parameter
    process_func = partial(process_single_image, output_folder=output_folder)
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_image = {executor.submit(process_func, img_path): img_path 
                          for img_path in image_files}
        
        # Collect results with progress bar
        for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_image), 
                               total=len(image_files), desc="Processing images"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                image_path = future_to_image[future]
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'filename': image_path,
                    'green_ratio': None,
                    'threshold': None,
                    'status': 'failed',
                    'error': str(e)
                })
    
    return results


#%% Main processing with parallel execution
if __name__ == "__main__":
    # Process images in parallel
    print("Starting parallel processing...")
    results = process_images_parallel(imfiles, imoutfolder, max_workers=None)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save initial results
    df.to_csv(os.path.join(imoutfolder, 'green_ratio_thresholds.csv'), index=False)
    print(f"Results saved to {os.path.join(imoutfolder, 'green_ratio_thresholds.csv')}")
    
    # Print processing summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    fail_count = sum(1 for r in results if r['status'] == 'failed')
    print(f"Processing complete: {success_count} successful, {fail_count} failed")

    #%% Statistical analysis to determine the optimal threshold
    # Filter successful results only
    df_analysis = df[df['status'] == 'success'].copy()
    df_analysis = df_analysis.dropna(subset=['threshold', 'green_ratio'])
    
    # Filter out thresholds <= 0.2 as they likely represent poor conditions
    df_analysis = df_analysis[df_analysis['threshold'] > 0.2]
    
    if len(df_analysis) == 0:
        print("No valid results for analysis!")
        exit()
    
    # Visualize threshold distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df_analysis['threshold'], bins=50, kde=True)
    plt.xlabel('Threshold')
    plt.ylabel('Frequency')
    plt.title('Distribution of GCC Thresholds (Tower RGB Images)')
    plt.savefig(os.path.join(imoutfolder, 'threshold_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== Initial Statistics ===")
    print(f"Valid images for analysis: {len(df_analysis)}")
    print(f"Mean threshold: {df_analysis['threshold'].mean():.4f}")
    print(f"Median threshold: {df_analysis['threshold'].median():.4f}")
    print(f"Std threshold: {df_analysis['threshold'].std():.4f}")
    print(f"Min threshold: {df_analysis['threshold'].min():.4f}")
    print(f"Max threshold: {df_analysis['threshold'].max():.4f}")
    
    #%% Remove outliers using IQR method
    print(f"\n=== Removing Outliers (IQR Method) ===")
    q1 = df_analysis['threshold'].quantile(0.25)
    q3 = df_analysis['threshold'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    print(f"Q1: {q1:.4f}, Q3: {q3:.4f}, IQR: {iqr:.4f}")
    print(f"IQR bounds: {lower_bound:.4f} to {upper_bound:.4f}")
    
    filtered_df_iqr = df_analysis[(df_analysis['threshold'] > lower_bound) & 
                                  (df_analysis['threshold'] < upper_bound)]
    print(f"Removed {len(df_analysis) - len(filtered_df_iqr)} outliers")
    
    print(f"\nAfter IQR filtering:")
    print(f"Mean threshold: {filtered_df_iqr['threshold'].mean():.4f}")
    print(f"Median threshold: {filtered_df_iqr['threshold'].median():.4f}")
    print(f"Std threshold: {filtered_df_iqr['threshold'].std():.4f}")
    
    #%% Remove outliers using MAD method (more robust)
    print(f"\n=== Removing Outliers (MAD Method) ===")
    filtered_df_mad = remove_outliers_mad(df_analysis, column='threshold')
    
    print(f"\nAfter MAD filtering:")
    print(f"Mean threshold: {filtered_df_mad['threshold'].mean():.4f}")
    print(f"Median threshold: {filtered_df_mad['threshold'].median():.4f}")
    print(f"Std threshold: {filtered_df_mad['threshold'].std():.4f}")
    
    #%% Visualize filtered results
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # IQR filtered
    axes[0].hist(filtered_df_iqr['threshold'], bins=30, alpha=0.7, edgecolor='black')
    axes[0].axvline(filtered_df_iqr['threshold'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {filtered_df_iqr["threshold"].mean():.4f}')
    axes[0].axvline(filtered_df_iqr['threshold'].median(), color='blue', linestyle='--', 
                    label=f'Median: {filtered_df_iqr["threshold"].median():.4f}')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('IQR Filtered Distribution')
    axes[0].legend()
    
    # MAD filtered
    axes[1].hist(filtered_df_mad['threshold'], bins=30, alpha=0.7, edgecolor='black')
    axes[1].axvline(filtered_df_mad['threshold'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {filtered_df_mad["threshold"].mean():.4f}')
    axes[1].axvline(filtered_df_mad['threshold'].median(), color='blue', linestyle='--', 
                    label=f'Median: {filtered_df_mad["threshold"].median():.4f}')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('MAD Filtered Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(imoutfolder, 'threshold_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    #%% Final recommendation
    print(f"\n{'='*60}")
    print(f"RECOMMENDED THRESHOLD FOR TOWER RGB IMAGES")
    print(f"{'='*60}")
    print(f"Mean threshold (MAD filtered): {filtered_df_mad['threshold'].mean():.4f}")
    print(f"Std threshold (MAD filtered): {filtered_df_mad['threshold'].std():.4f}")
    print(f"\nSuggested threshold value: {filtered_df_mad['threshold'].mean():.4f} ± {filtered_df_mad['threshold'].std():.4f}")
    print(f"\nUpdate the threshold in green_ratio_detector_GCC_towerRGB_ml.py:")
    print(f"  threshold = {filtered_df_mad['threshold'].mean():.4f}  # Tower RGB optimized")
    print(f"{'='*60}")
    
    # Save filtered results
    filtered_df_mad.to_csv(os.path.join(imoutfolder, 'green_ratio_thresholds_filtered.csv'), index=False)
    print(f"\nFiltered results saved to {os.path.join(imoutfolder, 'green_ratio_thresholds_filtered.csv')}")