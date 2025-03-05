#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian, threshold_otsu, sobel
from skimage.morphology import binary_erosion
from skimage import morphology
import os
import glob
import pandas as pd
#%%
def detect_frame_coordinates(img):
    """
    Automatically detects the metal frame coordinates in an image using contour detection.

    Args:
        image_path (str): Path to the image file.
        debug (bool): If True, shows intermediate processing steps.

    Returns:
        tuple: Coordinates of the frame (x1, y1, x2, y2), or None if frame detection fails.
    """
    try:

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = gaussian(gray, sigma=3) # cv2.GaussianBlur(gray, (3, 3), 0)

        edges = sobel(blurred)

        thresh = threshold_otsu(edges)
        binary = edges > thresh

        # Apply morphological operations to keep only straight lines
        kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size as needed
        result = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=2)
        result = binary_erosion(result)
        result = morphology.remove_small_objects(
            result,
            min_size=1000,
            connectivity=2
        )
        # plt.imshow(result, cmap='gray')
        y, x = np.where(result)

        if len(x) > 0:
            min_x, min_y = np.min(x), np.min(y)
            max_x, max_y = np.max(x), np.max(y)
            print(f"Bounding Box: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")
        else:
            print("No objects found in the binary image.")
            min_x, min_y, max_x, max_y = None, None, None, None

        return (min_x+100, min_y+100, max_x-100, max_y-100)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    
    
def quantify_vegetation(img, frame_coordinates):
    """
    Quantifies the ratio of green to non-green pixels within a specified frame in an image.
    Masks out pixels outside the frame.

    Args:
        image_path (str): Path to the image file.
        frame_coordinates (tuple): Coordinates of the frame (x1, y1, x2, y2).

    Returns:
        tuple: Ratio of green pixels to total pixels within the frame, and the green mask.
    """
    try:
        
        # Create a mask for the frame
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = frame_coordinates
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Apply the mask to the image
        masked_img = cv2.bitwise_and(img, img, mask=mask)
    
        # Extract the frame from the masked image
        frame = masked_img[y1:y2, x1:x2]

        # Convert the frame to the HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the green color range in HSV
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        # Create a mask for green pixels
        green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

        # Count the number of green pixels
        green_pixels = np.sum(green_mask > 0)

        # Count the total number of pixels in the frame
        total_pixels = frame.shape[0] * frame.shape[1]

        # Calculate the green ratio
        green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0

        return green_ratio, green_mask

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

#%%
imfolder = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/1_Data/1_Abisko/9_RGB_Close-up/1_CHMB/'
imfiles = glob.glob(imfolder + '**/*.JPG', recursive=True)
imoutfolder = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/1_Data/1_Abisko/9_RGB_Close-up/green_ratio/'
if not os.path.exists(imoutfolder):
    os.makedirs(imoutfolder)
#%%
data = []
for i in imfiles:
    print("processing: ", i)
    img = cv2.imread(i)
    frame_coordinates = detect_frame_coordinates(img)


    if frame_coordinates:
        print(f"Frame coordinates detected: {frame_coordinates}")

        # Quantify vegetation within the detected frame
        ratio, green_mask = quantify_vegetation(img, frame_coordinates)

        if ratio is not None:
            print(f"The green pixel ratio within the frame is: {ratio:.4f}")
            data.append({'filename': i, 'green_ratio': ratio})

            # Load the original image
            x1, y1, x2, y2 = frame_coordinates
            frame = img[y1:y2, x1:x2]

            # Apply the green mask to the frame
            masked_frame = cv2.bitwise_and(frame, frame, mask=green_mask)

            # Display the original image and the masked frame
            # Convert the images to RGB format for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            masked_frame_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)

            # Create a figure and axes
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Display the original image
            axes[0].imshow(img_rgb)
            axes[0].set_title("Original Image with Frame")
            axes[0].axis('off')  # Hide the axes
            axes[0].plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-', lw=2)  # Show the detected frame

            # Display the masked frame
            axes[1].imshow(masked_frame_rgb)
            axes[1].set_title("Green Masked Frame")
            axes[1].axis('off')  # Hide the axes

            plt.tight_layout()  # Adjust layout to prevent overlapping titles
            plt.show()

            # Save the figure
            fig.savefig(imoutfolder + i.split('/')[-1].replace('.JPG', '_green_masked_frame.JPG'))

        else:
            print("Vegetation quantification failed.")
            data.append({'filename': i, 'green_ratio': None})
    else:
        print("Frame detection failed.  Cannot quantify vegetation.")
        data.append({'filename': i, 'green_ratio': None})

df = pd.DataFrame(data)
df.to_csv(imoutfolder + 'green_ratio.csv', index=False, mode='w')
# %%
