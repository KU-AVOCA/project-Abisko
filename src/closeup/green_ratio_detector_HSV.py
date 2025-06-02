#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
#%%    
def quantify_vegetation(img):
    """
    Quantifies the ratio of green to non-green pixels in an image.

    Args:
        img: The input image

    Returns:
        tuple: Ratio of green pixels to total pixels, and the green mask.
    """
    try:
        # Convert the image to the HSV color space
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define the green color range in HSV
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        # Create a mask for green pixels
        green_mask = cv2.inRange(hsv_img, lower_green, upper_green)

        # Count the number of green pixels
        green_pixels = np.sum(green_mask > 0)

        # Count the total number of pixels in the image
        total_pixels = img.shape[0] * img.shape[1]

        # Calculate the green ratio
        green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0

        return green_ratio, green_mask

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

#%%
imfolder = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/1_Data/1_Abisko/9_RGB_Close-up/1_CHMB/0_ALL/'
imfiles = []
imfiles.extend(glob.glob(imfolder + '**/*.JPG', recursive=True))
imfiles.extend(glob.glob(imfolder + '**/*.jpg', recursive=True))
imfiles.extend(glob.glob(imfolder + '**/*.JPEG', recursive=True))
imfiles.extend(glob.glob(imfolder + '**/*.jpeg', recursive=True))
imoutfolder = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/1_Data/1_Abisko/9_RGB_Close-up/1_CHMB/0_ALL_green_ratio/'
if not os.path.exists(imoutfolder):
    os.makedirs(imoutfolder)
#%%
data = []
for i in imfiles:
    print("processing: ", i)
    img = cv2.imread(i)

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
        plt.show()

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
# %%
