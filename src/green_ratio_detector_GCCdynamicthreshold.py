'''
Green Ratio Detector for Vegetation Analysis
This script processes a collection of images to quantify vegetation coverage by calculating 
the ratio of green pixels to total pixels using the Greenness Chromatic Coordinate (GCC) index.
The script:
1. Recursively searches for images in the specified directory
2. Processes each image to detect green vegetation based on a GCC threshold
3. Calculates the green ratio (green pixels / total pixels)
4. Creates visual comparisons of original vs. green-masked images
5. Exports results to CSV and saves visualizations
Dependencies:
- OpenCV (cv2): For image processing
- NumPy: For numerical operations
- Matplotlib: For visualization
- Pandas: For data handling
- tqdm: For progress tracking
Usage:
- Set 'imfolder' to the directory containing images
- Set 'imoutfolder' to the desired output directory
- Adjust the Greenness threshold in quantify_vegetation() if needed (currently 0.36)
- Run the script to process all images and generate results
Output:
- Comparative visualizations of original and green-masked images
- CSV file with green ratio values for all processed images

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
# import datetime
# import re
sns.set_theme(style="darkgrid", font_scale=1.5)
#%%
imfolder = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/1_Data/1_Abisko/9_RGB_Close-up/1_CHMB/0_ALL/'
imfiles = []
imfiles.extend(glob.glob(imfolder + '**/*.JPG', recursive=True))
imfiles.extend(glob.glob(imfolder + '**/*.jpg', recursive=True))
imfiles.extend(glob.glob(imfolder + '**/*.JPEG', recursive=True))
imfiles.extend(glob.glob(imfolder + '**/*.jpeg', recursive=True))
imoutfolder = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/1_Data/1_Abisko/9_RGB_Close-up/1_CHMB/0_ALL_green_ratio_determinethreshold/'
if not os.path.exists(imoutfolder):
    os.makedirs(imoutfolder)
#%%    
def quantify_vegetation(img, threshold):
    """
    Quantifies the ratio of green to non-green pixels using Greenness index (GCC).

    Args:
        img: The input image (BGR format)

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
        # Note: Adding a small value to prevent division by zero
        greenness = g / (r + g + b + 1e-10) #  

        # Create a binary mask using a threshold (adjust as needed)
        # threshold = 0.36  # don't ask me why this value
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
    Filters out pixels with GCC < 0.2 as they represent poor light conditions
    or non-vegetation areas.
    
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
    
    # Filter out pixels with low GCC (< 0.2)
    filtered_greenness = greenness.copy()
    filtered_greenness[greenness < 0.2] = 0
    
    # Scale to 0-255 for Otsu
    greenness_scaled = (filtered_greenness * 255).astype(np.uint8)
    
    # Apply Otsu's method
    threshold_value, _ = cv2.threshold(greenness_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert back to original GCC scale (0-1)
    threshold = threshold_value / 255.0
    
    # Ensure threshold is at least 0.2
    threshold = max(threshold, 0.2)
    
    return threshold

def determine_threshold_kmeans(img, n_clusters):
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
    # With 2 clusters, use a point between them
    lower_center = centers[sorted_indices[0]][0]
    higher_center = centers[sorted_indices[1]][0]
    
    # Use a weighted position between clusters
    # Slightly closer to the lower center to avoid false positives
    threshold = lower_center + 0.6 * (higher_center - lower_center)
    
    return threshold

#%%
data = []
for i in tqdm.tqdm(imfiles, desc="Processing images"):
    img = cv2.imread(i)
    print("processing: ", i)

    # Quantify vegetation within the whole image
    threshold = determine_threshold_otsu(img)
    # threshold = determine_threshold_kmeans(img, n_clusters=2)
    print(f"Threshold determined: {threshold:.4f}")
    ratio, green_mask = quantify_vegetation(img, threshold)

    if ratio is not None:
        print(f"The green pixel ratio is: {ratio:.4f}")
        data.append({'filename': i, 'green_ratio': ratio, 'threshold': threshold})

        # Apply the green mask to the image
        masked_img = cv2.bitwise_and(img, img, mask=green_mask)

        # Display the original image and the masked image
        # Convert the images to RGB format for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masked_img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)

        # Create a figure and axes
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Display the original image
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis('off')  # Hide the axes

        # Display the masked image
        axes[1].imshow(masked_img_rgb)
        axes[1].set_title("Green Masked Image")
        axes[1].axis('off')  # Hide the axes

        plt.tight_layout()  # Adjust layout to prevent overlapping titles
        # plt.show()

        # Save the figure
        # Extract base filename and extension
        base_filename = os.path.splitext(os.path.basename(i))[0]
        extension = os.path.splitext(os.path.basename(i))[1]
        # Save the figure
        fig.savefig(os.path.join(imoutfolder, f"{base_filename}_green_masked{extension}"))

    else:
        print("Vegetation quantification failed.")
        data.append({'filename': i, 'green_ratio': None, 'threshold': None})

df = pd.DataFrame(data)
df.to_csv(imoutfolder + 'green_ratio.csv', index=False, mode='w')
print(f"Done! Results saved to {imoutfolder + 'green_ratio.csv'}")

#%% statistical analysis to determine the threshold
df = pd.read_csv(imoutfolder + 'green_ratio.csv')
df = df.dropna(subset=['threshold', 'green_ratio'])
# Filter out thresholds <= 0.2 as they likely represent non-vegetation
# df = df[df['threshold'] > 0.2]
sns.histplot(df['threshold'], bins=20)
#%% remove outliers
# q1 = df['threshold'].quantile(0.25)
# q3 = df['threshold'].quantile(0.75)
# iqr = q3 - q1
# lower_bound = q1 - 1.5 * iqr
# upper_bound = q3 + 1.5 * iqr
# df = df[(df['threshold'] > lower_bound) & (df['threshold'] < upper_bound)]
# sns.histplot(df['threshold'], bins=20)
# print(f"the mean threshold is {df['threshold'].mean():.4f}")
# print(f"the median threshold is {df['threshold'].median():.4f}")
# print(f"the std threshold is {df['threshold'].std():.4f}")
# %% remove outliers using MAD
def remove_outliers_mad(df, column='threshold', threshold_factor=2.5):
    """
    Remove outliers using Median Absolute Deviation from scipy.stats - more robust than IQR
    """
    
    median = df[column].median()
    # Calculate MAD using scipy.stats
    mad = stats.median_abs_deviation(df[column], scale="normal")
    
    # Scale factor for normal distribution (1.4826 for normal distribution)
    mad_scaled = mad #* 1.4826 is already assumed by setting scale="normal"
    
    # Define bounds
    lower_bound = median - threshold_factor * mad_scaled
    upper_bound = median + threshold_factor * mad_scaled
    
    print(f"MAD bounds: {lower_bound:.4f} to {upper_bound:.4f}")
    
    # Filter dataframe
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    print(f"Removed {len(df) - len(filtered_df)} outliers using MAD method")
    
    return filtered_df

# Usage:
df_filtered_mad = remove_outliers_mad(df)
sns.histplot(df_filtered_mad['threshold'], bins=20, color='green')
plt.title('Threshold Distribution after MAD Filtering')

print(f"Mean threshold (MAD): {df_filtered_mad['threshold'].mean():.4f}")
print(f"Median threshold (MAD): {df_filtered_mad['threshold'].median():.4f}")
print(f"Std threshold (MAD): {df_filtered_mad['threshold'].std():.4f}")
# %% visualize the results
# df = pd.read_csv(imoutfolder + 'green_ratio.csv')
# df['imname'] = df.filename.apply(lambda x: os.path.basename(x))
# # Extract datetime from filenames with pattern like "07-07-2022_E2.JPG"

# # Regular expression pattern for "DD-MM-YYYY" format
# date_pattern = re.compile(r'(\d{2})-(\d{2})-(\d{4})_')

# # Function to extract datetime from filename
# def extract_datetime(filename):
#     match = date_pattern.search(filename)
#     if match:
#         day, month, year = match.groups()
#         try:
#             return datetime.datetime(int(year), int(month), int(day))
#         except ValueError:
#             return None
#     return None

# # Apply the function to create a datetime column
# df['datetime'] = df['imname'].apply(extract_datetime)

# # Filter dataframe to only include rows with valid datetime
# df_with_dates = df.dropna(subset=['datetime'])

# # Print information about files with dates
# if not df_with_dates.empty:
#     print(f"Found {len(df_with_dates)} images with date information in the filename")
#     print(f"Date range: {df_with_dates['datetime'].min().date()} to {df_with_dates['datetime'].max().date()}")
    
#     # Display the first few entries with dates
#     print("\nSample of images with dates:")
#     print(df_with_dates[['imname', 'datetime', 'green_ratio']].head())
# else:
#     print("No images with date information in the filename were found")
# # %%
# df = df.dropna(subset='datetime')
# df['year'] = df['datetime'].dt.year

# sns.boxplot(x='year', y='green_ratio', data=df)
# plt.title("Green Ratio by Year")
# plt.ylabel("Green Ratio")
# plt.xlabel("Year")

# #%%
# df['doy'] = df['datetime'].dt.dayofyear
# sns.lineplot(x='doy', y='green_ratio', data=df, hue='year')
# # %%
