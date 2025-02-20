#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

#%%
def detect_frame_and_crop(image_path, output_path='cropped_image_improved.jpg', expansion_pixels=10):
    """
    Detects a rectangular frame and crops the image, with improved contour approximation and bounding box expansion.

    Args:
        image_path (str): Path to the input image file.
        output_path (str, optional): Path to save the cropped output image. Defaults to 'cropped_image_improved.jpg'.
        expansion_pixels (int, optional): Number of pixels to expand the bounding box by in each direction. Defaults to 10.
    """
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not open or read image file: {image_path}")
        return

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # 4. Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_contour = None
    max_area = 0

    # 5. Find the frame contour (largest rectangle)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            perimeter = cv2.arcLength(contour, True)
            # Loosen contour approximation by reducing epsilon factor (e.g., 0.02)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) > 4:
                if area > max_area:
                    max_area = area
                    frame_contour = approx

    if frame_contour is not None:
        # 6. Get bounding box
        x, y, w, h = cv2.boundingRect(frame_contour)

        # 7. Expand the bounding box
        x_start = max(0, x - expansion_pixels)  # Ensure start is not negative
        y_start = max(0, y - expansion_pixels)
        x_end = min(img.shape[1], x + w + expansion_pixels) # Ensure end is within image width
        y_end = min(img.shape[0], y + h + expansion_pixels) # Ensure end is within image height


        # 8. Crop the image using expanded bounding box
        cropped_image = img[y_start:y_end, x_start:x_end]

        # 8. display the origional and cropped image
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        ax[1].imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        ax[1].set_title('Cropped Image')
        ax[1].axis('off')
        plt.show()
        # cv2.imwrite(output_path, cropped_image)
        # print(f"Cropped image saved to {output_path}")

        # Optional: Display the cropped image (for testing purposes)
        # cv2.imshow('Cropped Image', cropped_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("No frame detected in the image.")

# Example usage:
image_file_path = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/1_Data/1_Abisko/9_RGB_Close-up/1_CHMB/2_2022/20220622_G1.JPG'  # Replace 'image.jpg' with the actual path to your image file
detect_frame_and_crop(image_file_path)
# %%
