
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
# import datetime
# import re
sns.set_theme(style="darkgrid", font_scale=1.5)
#%%
imfolder1 = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/1_Simon/1_Abisko/4_CHMB_RGB/3_2023/2_Work/2_Cropped/'
imfiles = []
imfiles.extend(glob.glob(imfolder1 + '**/*.JPG', recursive=True))
imfiles.extend(glob.glob(imfolder1 + '**/*.jpg', recursive=True))
imfiles.extend(glob.glob(imfolder1 + '**/*.JPEG', recursive=True))
imfiles.extend(glob.glob(imfolder1 + '**/*.jpeg', recursive=True))   

imfolder2 = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/1_Simon/1_Abisko/4_CHMB_RGB/2_2022/2_Work/2_Cropped/'
imfiles.extend(glob.glob(imfolder2 + '**/*.JPG', recursive=True))
imfiles.extend(glob.glob(imfolder2 + '**/*.jpg', recursive=True))
imfiles.extend(glob.glob(imfolder2 + '**/*.JPEG', recursive=True))
imfiles.extend(glob.glob(imfolder2 + '**/*.jpeg', recursive=True))

print(f"Found {len(imfiles)} images in total")

imoutfolder = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/closeup_green_ratio_mean/'
if not os.path.exists(imoutfolder):
    os.makedirs(imoutfolder)
#%%    
def quantify_vegetation(img):
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
        threshold = 0.38 # 0.37 +- 0.07
        green_mask = (greenness > threshold).astype(np.uint8) * 255
        
        # Count green pixels and calculate ratio
        green_pixels = np.sum(green_mask > 0)
        total_pixels = img.shape[0] * img.shape[1]
        green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0
        
        return green_ratio, green_mask

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None



#%%
data = []
for i in tqdm.tqdm(imfiles, desc="Processing images"):
    img = cv2.imread(i)
    print("processing: ", i)

    # Quantify vegetation within the whole image
    ratio, green_mask = quantify_vegetation(img)

    if ratio is not None:
        print(f"The green pixel ratio is: {ratio:.4f}")
        data.append({'filename': i, 'green_ratio': ratio})

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
        data.append({'filename': i, 'green_ratio': None})

df = pd.DataFrame(data)
df.to_csv(imoutfolder + 'green_ratio.csv', index=False, mode='w')
print(f"Done! Results saved to {imoutfolder + 'green_ratio.csv'}")
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
