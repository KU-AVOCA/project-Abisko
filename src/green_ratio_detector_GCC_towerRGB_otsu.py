'''
Green Ratio Detector for Vegetation Analysis
This script processes a collection of images to quantify vegetation coverage by calculating 
the ratio of green pixels to total pixels using the Greenness Chromatic Coordinate (GCC) index.
The script:
1. Recursively searches for images in the specified directory
2. Extracts datetime metadata from each image
3. Processes each image to detect green vegetation based on a GCC threshold
4. Calculates the green ratio (green pixels / total pixels)
5. Creates visual comparisons of original vs. green-masked images
6. Exports results to CSV incrementally with datetime info and saves visualizations
Dependencies:
- OpenCV (cv2): For image processing
- NumPy: For numerical operations
- Matplotlib: For visualization
- Pandas: For data handling
- tqdm: For progress tracking
- PIL: For extracting image metadata
Usage:
- Set 'imfolder' to the directory containing images
- Set 'imoutfolder' to the desired output directory
- Adjust the Greenness threshold in quantify_vegetation() if needed (currently 0.36)
- Run the script to process all images and generate results
Output:
- Comparative visualizations of original and green-masked images (preserving folder structure)
- CSV file with green ratio values and date/time information for all processed images

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
import datetime
import re
from PIL import Image
from PIL.ExifTags import TAGS

sns.set_theme(style="darkgrid", font_scale=1.5)
#%%
imfolder = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/1_Simon/1_Abisko/6_Tower_Data/Tower RGB images/1 Data/1 Years'
imfiles = []
imfiles.extend(glob.glob(os.path.join(imfolder, '**/', '*.JPG'), recursive=True))
imfiles.extend(glob.glob(os.path.join(imfolder, '**/', '*.jpg'), recursive=True))
imfiles.extend(glob.glob(os.path.join(imfolder, '**/', '*.JPEG'), recursive=True))
imfiles.extend(glob.glob(os.path.join(imfolder, '**/', '*.jpeg'), recursive=True))
imfiles.extend(glob.glob(os.path.join(imfolder, '**/', '*.png'), recursive=True))
imfiles.extend(glob.glob(os.path.join(imfolder, '**/', '*.PNG'), recursive=True))
print(f"Found {len(imfiles)} images in {imfolder}")
imoutfolder = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/1_Simon/1_Abisko/6_Tower_Data/Tower RGB images/1_Data_greenessByShunan_Otsu'
if not os.path.exists(imoutfolder):
    os.makedirs(imoutfolder)
#%%
# Create the main results directory if it doesn't exist
results_dir = os.path.join(imoutfolder, 'results')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Initialize the CSV file with headers
csv_path = os.path.join(results_dir, 'green_ratio.csv')
with open(csv_path, 'w') as f:
    f.write('filename,datetime,green_ratio,class1_ratio,class2_ratio\n')
#%%
def get_image_datetime(image_path):
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
def quantify_vegetation(img):
    """
    Quantifies the ratio of green to non-green pixels using Greenness index (GCC).
    Also separates green vegetation into two classes (likely trees and understory).

    Args:
        img: The input image (BGR format)

    Returns:
        tuple: Overall green ratio, green mask, class1 ratio, class2 ratio, class visualization
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
        threshold = 0.36  # don't ask me why this value
        green_mask = (greenness > threshold).astype(np.uint8) * 255
        
        # Count green pixels and calculate ratio
        green_pixels = np.sum(green_mask > 0)
        total_pixels = img.shape[0] * img.shape[1]
        green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0
        
        # ENHANCEMENT: Separate green vegetation into two classes
        
        # Step 1: Create a masked green-only image (preserving original values)
        masked_green = cv2.bitwise_and(img, img, mask=(green_mask // 255).astype(np.uint8))
        
        # Step 2: Convert masked image to grayscale for Otsu thresholding
        masked_gray = cv2.cvtColor(masked_green, cv2.COLOR_BGR2GRAY)
        
        # Step 3: Apply Otsu's thresholding to separate two classes of vegetation
        # Note: We only apply this to non-zero pixels (actual green vegetation)
        non_zero_mask = masked_gray > 0
        if np.sum(non_zero_mask) > 0:  # Check if there are green pixels
            pixels_for_otsu = masked_gray[non_zero_mask]
            otsu_threshold, _ = cv2.threshold(pixels_for_otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply the threshold to separate vegetation classes
            class1_mask = np.logical_and(masked_gray > 0, masked_gray < otsu_threshold)
            class2_mask = np.logical_and(masked_gray > 0, masked_gray >= otsu_threshold)
            
            # Calculate individual class ratios
            class1_pixels = np.sum(class1_mask)
            class2_pixels = np.sum(class2_mask)
            
            class1_ratio = class1_pixels / total_pixels if total_pixels > 0 else 0
            class2_ratio = class2_pixels / total_pixels if total_pixels > 0 else 0
            
            # Create a visualization of the two classes
            visualization = np.zeros_like(img)
            # Class 1 - e.g., understory (shown in green)
            visualization[class1_mask] = [0, 255, 0]
            # Class 2 - e.g., trees (shown in blue) 
            visualization[class2_mask] = [255, 0, 0]
        else:
            class1_ratio = 0
            class2_ratio = 0
            visualization = np.zeros_like(img)
        
        return green_ratio, green_mask, class1_ratio, class2_ratio, visualization

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None, None

#%%
# Process each image and write results immediately to CSV
for i in tqdm.tqdm(imfiles, desc="Processing images"):
    img = cv2.imread(i)
    if img is None:
        print(f"Could not read image: {i}")
        continue
        
    print("processing: ", i)

    # Get image datetime from EXIF metadata
    image_datetime = get_image_datetime(i)
    datetime_str = image_datetime if image_datetime else "NA"
    print(f"Image datetime: {datetime_str}")
    
    # Quantify vegetation within the whole image
    ratio, green_mask, class1_ratio, class2_ratio, class_vis = quantify_vegetation(img)

    if ratio is not None:
        print(f"The green pixel ratio is: {ratio:.4f}")
        print(f"Class 1 (likely understory) ratio: {class1_ratio:.4f}")
        print(f"Class 2 (likely trees) ratio: {class2_ratio:.4f}")
        
        # Write result to CSV immediately
        with open(csv_path, 'a') as f:
            f.write(f'{i},{datetime_str},{ratio},{class1_ratio},{class2_ratio}\n')

        # Apply the green mask to the image
        masked_img = cv2.bitwise_and(img, img, mask=green_mask // 255)

        # Display the original image, the masked image, and vegetation classes
        # Convert the images to RGB format for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masked_img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
        class_vis_rgb = cv2.cvtColor(class_vis, cv2.COLOR_BGR2RGB)

        # Create a figure and axes
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Display the original image
        axes[0].imshow(img_rgb)
        title_text = f"Original ({datetime_str})" if datetime_str != "NA" else "Original"
        axes[0].set_title(title_text)
        axes[0].axis('off')  # Hide the axes

        # Display the masked image
        axes[1].imshow(masked_img_rgb)
        axes[1].set_title(f"Green Masked (GR={ratio:.2f})")
        axes[1].axis('off')  # Hide the axes
        
        # Display the vegetation classes
        axes[2].imshow(class_vis_rgb)
        axes[2].set_title(f"Vegetation Classes\nClass1={class1_ratio:.2f}, Class2={class2_ratio:.2f}")
        axes[2].axis('off')

        plt.tight_layout()  # Adjust layout to prevent overlapping titles

        # Create output directory structure that mirrors input
        rel_path = os.path.relpath(os.path.dirname(i), imfolder)
        output_dir = os.path.join(imoutfolder, rel_path)
        
        # Create the directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Extract base filename and extension
        base_filename = os.path.splitext(os.path.basename(i))[0]
        extension = os.path.splitext(os.path.basename(i))[1]
        
        # Save the figure to the mirrored directory structure
        output_path = os.path.join(output_dir, f"{base_filename}_green_masked{extension}")
        fig.savefig(output_path)
        plt.close(fig)  # Close the figure to free memory

    else:
        print(f"Vegetation quantification failed for {i}")
        # Write failure to CSV
        with open(csv_path, 'a') as f:
            f.write(f'{i},{datetime_str},NA,NA,NA\n')

print(f"Done! Results saved to {csv_path}")

#%% under all subfolders
# Load the CSV file and visualize the results
# df = pd.read_csv('/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/1_Simon/1_Abisko/6_Tower_Data/Tower RGB images/1_Data_greenessByShunan/results/green_ratio.csv',
#                  nrows=8680)
# df['datetime'] = pd.to_datetime(df['datetime'])
# df['year'] = df['datetime'].dt.year
# df['doys'] = df['datetime'].dt.dayofyear
# df['imgroup'] = df['filename'].str.split('/').str[-2]
# # %%
# fig, ax = plt.subplots(figsize=(12, 6))
# sns.lineplot(data=df, x='doys', y='green_ratio', hue='imgroup', ax=ax)
# ax.set(xlabel='Day of Year', ylabel='Green Ratio', title='Green Ratio Over Time')
# # %%
# fig, ax = plt.subplots(figsize=(12, 6))
# sns.lineplot(data=df, x='doys', y='green_ratio', hue='year', ax=ax)
# ax.set(xlabel='Day of Year', ylabel='Green Ratio', title='Green Ratio Over Time')
# # %%
# fig, ax = plt.subplots(figsize=(12, 6))
# sns.boxplot(data=df, x='year', y='green_ratio', ax=ax)
# ax.set(xlabel='Year', ylabel='Green Ratio', title='Green Ratio Distribution by Year')
# # %% under 1 Years folder
# # Load the CSV file and visualize the results
# df = pd.read_csv('/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/1_Simon/1_Abisko/6_Tower_Data/Tower RGB images/1_Data_greenessByShunan/results/green_ratio.csv',
#                  header=0, skiprows=range(1, 8681))
# df['datetime'] = pd.to_datetime(df['datetime'])
# df['year'] = df['datetime'].dt.year
# df['doys'] = df['datetime'].dt.dayofyear
# df['imgroup'] = df['filename'].str.split('/').str[-3]
# # %%
# fig, ax = plt.subplots(figsize=(12, 6))
# sns.lineplot(data=df, x='doys', y='green_ratio', hue='imgroup', ax=ax)
# ax.set(xlabel='Day of Year', ylabel='Green Ratio', title='Green Ratio Over Time')

# # %%
# # Create a figure with subplots for each imgroup
# unique_imgroups = df['imgroup'].unique()
# fig, axs = plt.subplots(len(unique_imgroups), 1, figsize=(14, 10), sharex=True)

# # Plot data for each imgroup in separate subplots
# handles, labels = None, None

# for i, imgroup in enumerate(unique_imgroups):
#     group_data = df[df['imgroup'] == imgroup]
#     g = sns.lineplot(data=group_data, x='doys', y='green_ratio', hue='year', ax=axs[i])
    
#     # Store handles and labels from the first plot to use for the shared legend
#     if i == 0:
#         handles, labels = axs[i].get_legend_handles_labels()
    
#     # Remove individual legends
#     axs[i].get_legend().remove()
    
#     axs[i].set_ylabel('Green Ratio')
#     axs[i].set_title(f'Green Ratio Over Time - {imgroup}')
    
#     # Only set xlabel for the bottom subplot
#     if i == len(unique_imgroups) - 1:
#         axs[i].set_xlabel('Day of Year')
#     else:
#         axs[i].set_xlabel('')

# # Add a single legend for all subplots
# fig.legend(handles, labels, title="Year", loc='upper right', bbox_to_anchor=(1.15, 0.9))

# plt.tight_layout()
# # %%
# fig, ax = plt.subplots(figsize=(12, 6))
# sns.boxplot(data=df, x='year', y='green_ratio', ax=ax, hue='imgroup')
# ax.set(xlabel='Year', ylabel='Green Ratio', title='Green Ratio Distribution by Year')

# # %%
