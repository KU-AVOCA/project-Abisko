'''
Dynamic Threshold Detector for Tower RGB Images

This script processes tower RGB images to determine an optimal GCC threshold
by calculating thresholds for individual images and analyzing their distribution.

Based on green_ratio_detector_GCCdynamicthreshold.py but adapted for tower images.

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


def remove_outliers_mad(df, column='threshold', threshold_factor=3.5):
    """
    Remove outliers using Median Absolute Deviation (MAD) method.
    More robust than IQR for skewed distributions.
    
    Args:
        df: DataFrame with threshold values
        column: Column name containing threshold values
        threshold_factor: Number of MADs to use as cutoff (default: 3.5)
        
    Returns:
        DataFrame: Filtered DataFrame without outliers
    """
    median = df[column].median()
    mad = np.median(np.abs(df[column] - median))
    mad_scaled = mad * 1.4826  # Scale factor for normal distribution
    
    # Define bounds
    lower_bound = median - threshold_factor * mad_scaled
    upper_bound = median + threshold_factor * mad_scaled
    
    print(f"MAD bounds: {lower_bound:.4f} to {upper_bound:.4f}")
    
    # Filter dataframe
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    print(f"Removed {len(df) - len(filtered_df)} outliers using MAD method")
    
    return filtered_df

#%%
# Sample a subset of images for threshold determination (to speed up processing)
# You can adjust sample_size or set to None to process all images
sample_size = 500  # Set to None to process all images
if sample_size and len(imfiles) > sample_size:
    import random
    random.seed(42)
    imfiles_sample = random.sample(imfiles, sample_size)
    print(f"Processing {len(imfiles_sample)} sampled images")
else:
    imfiles_sample = imfiles
    print(f"Processing all {len(imfiles_sample)} images")

#%%
data = []
for i in tqdm.tqdm(imfiles_sample, desc="Processing images"):
    img = cv2.imread(i)
    if img is None:
        print(f"Could not read image: {i}")
        continue
    
    print("processing: ", i)

    # Determine threshold using Otsu's method
    threshold = determine_threshold_otsu(img)
    # Alternative: use K-means
    # threshold = determine_threshold_kmeans(img, n_clusters=2)
    
    print(f"Threshold determined: {threshold:.4f}")
    ratio, green_mask = quantify_vegetation(img, threshold)

    if ratio is not None:
        print(f"The green pixel ratio is: {ratio:.4f}")
        data.append({'filename': i, 'green_ratio': ratio, 'threshold': threshold})
    else:
        print("Vegetation quantification failed.")
        data.append({'filename': i, 'green_ratio': None, 'threshold': None})

df = pd.DataFrame(data)
df.to_csv(imoutfolder + 'green_ratio_thresholds.csv', index=False, mode='w')
print(f"Done! Results saved to {imoutfolder + 'green_ratio_thresholds.csv'}")

#%% Statistical analysis to determine the optimal threshold
df = pd.read_csv(imoutfolder + 'green_ratio_thresholds.csv')
df = df.dropna(subset=['threshold', 'green_ratio'])

# Filter out thresholds <= 0.2 as they likely represent poor conditions
df = df[df['threshold'] > 0.2]

# Visualize threshold distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['threshold'], bins=50, kde=True)
plt.xlabel('Threshold')
plt.ylabel('Frequency')
plt.title('Distribution of GCC Thresholds (Tower RGB Images)')
plt.savefig(imoutfolder + 'threshold_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n=== Initial Statistics ===")
print(f"Mean threshold: {df['threshold'].mean():.4f}")
print(f"Median threshold: {df['threshold'].median():.4f}")
print(f"Std threshold: {df['threshold'].std():.4f}")
print(f"Min threshold: {df['threshold'].min():.4f}")
print(f"Max threshold: {df['threshold'].max():.4f}")

#%% Remove outliers using IQR method
print(f"\n=== Removing Outliers (IQR Method) ===")
q1 = df['threshold'].quantile(0.25)
q3 = df['threshold'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print(f"Q1: {q1:.4f}, Q3: {q3:.4f}, IQR: {iqr:.4f}")
print(f"IQR bounds: {lower_bound:.4f} to {upper_bound:.4f}")

filtered_df_iqr = df[(df['threshold'] > lower_bound) & (df['threshold'] < upper_bound)]
print(f"Removed {len(df) - len(filtered_df_iqr)} outliers")

print(f"\nAfter IQR filtering:")
print(f"Mean threshold: {filtered_df_iqr['threshold'].mean():.4f}")
print(f"Median threshold: {filtered_df_iqr['threshold'].median():.4f}")
print(f"Std threshold: {filtered_df_iqr['threshold'].std():.4f}")

#%% Remove outliers using MAD method (more robust)
print(f"\n=== Removing Outliers (MAD Method) ===")
filtered_df_mad = remove_outliers_mad(df, column='threshold', threshold_factor=3.5)

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
plt.savefig(imoutfolder + 'threshold_comparison.png', dpi=300, bbox_inches='tight')
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
filtered_df_mad.to_csv(imoutfolder + 'green_ratio_thresholds_filtered.csv', index=False)
print(f"\nFiltered results saved to {imoutfolder + 'green_ratio_thresholds_filtered.csv'}")