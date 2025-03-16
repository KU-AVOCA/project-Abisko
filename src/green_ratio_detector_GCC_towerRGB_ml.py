'''
Enhanced Green Ratio Detector for Vegetation Analysis with Machine Learning Classification

This script processes a collection of time-lapse images to quantify vegetation coverage and 
separate different vegetation types (likely trees and understory plants) using advanced 
machine learning techniques.

The script:
1. Recursively searches for images in the specified directory
2. Extracts datetime metadata from each image
3. Processes each image to detect green vegetation based on a GCC threshold
4. Separates green vegetation into two classes using a selected ML method:
   - K-means clustering (default)
   - Gaussian Mixture Models (GMM)
   - DBSCAN spatial clustering
   - Spectral clustering
5. Calculates ratios for total green vegetation and each vegetation class
6. Creates visual comparisons showing the original image, green mask, and classified vegetation
7. Exports results to CSV with datetime info and saves visualizations

Dependencies:
- OpenCV (cv2): For image processing
- NumPy: For numerical operations
- Matplotlib: For visualization
- scikit-learn: For machine learning algorithms
- tqdm: For progress tracking
- PIL: For extracting image metadata

Usage:
- Set 'imfolder' to the directory containing images
- Set 'imoutfolder' to the desired output directory
- Set 'classification_method' to one of: "kmeans", "gmm", "dbscan", "spectral"
- Run the script to process all images and generate results

Output:
- Comparative visualizations with original, green-masked, and classified images
- CSV file with green ratio values and class-specific ratios for all processed images

Shunan Feng (shf@ign.ku.dk)
'''
#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import glob
import tqdm
import seaborn as sns
import datetime
from PIL import Image
from PIL.ExifTags import TAGS

# Machine learning imports
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

sns.set_theme(style="darkgrid", font_scale=1.5)

#%%
# Set the classification method to use
classification_method = "kmeans"  # Options: "kmeans", "gmm", "dbscan", "spectral"

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

imoutfolder = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/Tower RGB images/Data_greenessByShunan_' + classification_method
if not os.path.exists(imoutfolder):
    os.makedirs(imoutfolder)
#%%
# Create the main results directory if it doesn't exist
results_dir = os.path.join(imoutfolder, 'results')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Initialize the CSV file with headers
csv_path = os.path.join(results_dir, 'green_ratio' + classification_method + '.csv')
with open(csv_path, 'w') as f:
    f.write('filename,datetime,green_ratio,green_mean,green_std,green_norm,class1_ratio,class1_mean,class1_std,class1_norm,class2_ratio,class2_mean,class2_std,class2_norm,method\n')
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
def quantify_vegetation_kmeans(img):
    """
    Quantifies green vegetation using K-means clustering.
    
    This method first identifies green pixels using GCC index, then separates them
    into two classes (likely trees and understory) using K-means clustering in LAB color space.
    
    Args:
        img: The input image (BGR format)

    Returns:
        tuple: Overall green metrics (ratio, mean, std, norm_greenness),
               green mask, 
               class1 metrics (ratio, mean, std, norm_greenness), 
               class2 metrics (ratio, mean, std, norm_greenness), 
               class visualization
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
        greenness = g / (r + g + b + 1e-10)

        # Create a binary mask using a threshold (adjust as needed)
        threshold = 0.37 # 0.37 +- 0.07
        green_mask = (greenness > threshold).astype(np.uint8) * 255
        
        # Count green pixels and calculate ratio
        green_pixels = np.sum(green_mask > 0)
        total_pixels = img.shape[0] * img.shape[1]
        green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0
        
        # Calculate mean and std greenness for all green pixels
        green_pixels_mask = green_mask > 0
        if green_pixels > 0:
            mean_greenness = np.mean(greenness[green_pixels_mask])
            std_greenness = np.std(greenness[green_pixels_mask])
            norm_greenness = np.sum(greenness[green_pixels_mask]) / green_pixels
        else:
            mean_greenness = 0
            std_greenness = 0
            norm_greenness = 0
            
        # Store overall green metrics as a dictionary
        green_metrics = {
            'ratio': green_ratio,
            'mean': mean_greenness,
            'std': std_greenness,
            'norm_greenness': norm_greenness
        }
        
        # ENHANCED CLASSIFICATION: Using K-means for vegetation class separation
        
        # Step 1: Create a masked green-only image
        masked_green = cv2.bitwise_and(img, img, mask=(green_mask // 255).astype(np.uint8))
        
        # Step 2: Prepare data for K-means clustering - only include green pixels
        non_zero_mask = np.any(masked_green != 0, axis=2)
        
        # Initialize class metrics
        class1_metrics = {'ratio': 0, 'mean': 0, 'std': 0, 'norm_greenness': 0}
        class2_metrics = {'ratio': 0, 'mean': 0, 'std': 0, 'norm_greenness': 0}
        visualization = np.zeros_like(img)
        
        if np.sum(non_zero_mask) > 0:  # Check if there are green pixels
            # Extract features for clustering (using multiple channels for better separation)
            # Convert to LAB color space which is better for color-based segmentation
            lab_image = cv2.cvtColor(masked_green, cv2.COLOR_BGR2LAB)
            
            # Reshape to a list of pixels with features
            pixels = lab_image[non_zero_mask].reshape(-1, 3).astype(np.float32)
            
            # Apply K-means clustering (K=2 for trees and understory)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Create masks for each class
            # First, create a full-size labels array initialized with -1 (no class)
            full_labels = np.full(img.shape[:2], -1, dtype=int)
            # Then, set the labels only for green pixels
            full_labels[non_zero_mask] = labels.flatten()
            
            # Determine which cluster is likely understory (darker) and which is trees (brighter)
            if centers[0][0] < centers[1][0]:  # Lower L value = darker
                class1_mask = (full_labels == 0)  # Understory (darker)
                class2_mask = (full_labels == 1)  # Trees (brighter)
            else:
                class1_mask = (full_labels == 1)  # Understory (darker)
                class2_mask = (full_labels == 0)  # Trees (brighter)
            
            # Calculate individual class ratios
            class1_pixels = np.sum(class1_mask)
            class2_pixels = np.sum(class2_mask)
            
            class1_ratio = class1_pixels / total_pixels if total_pixels > 0 else 0
            class2_ratio = class2_pixels / total_pixels if total_pixels > 0 else 0
            
            # Calculate mean, std, and normalized greenness for each class
            if class1_pixels > 0:
                class1_mean = np.mean(greenness[class1_mask])
                class1_std = np.std(greenness[class1_mask])
                class1_norm_greenness = np.sum(greenness[class1_mask]) / class1_pixels
            else:
                class1_mean = 0
                class1_std = 0
                class1_norm_greenness = 0
                
            if class2_pixels > 0:
                class2_mean = np.mean(greenness[class2_mask])
                class2_std = np.std(greenness[class2_mask])
                class2_norm_greenness = np.sum(greenness[class2_mask]) / class2_pixels
            else:
                class2_mean = 0
                class2_std = 0
                class2_norm_greenness = 0
                
            # Store class metrics
            class1_metrics = {
                'ratio': class1_ratio,
                'mean': class1_mean,
                'std': class1_std,
                'norm_greenness': class1_norm_greenness
            }
            
            class2_metrics = {
                'ratio': class2_ratio,
                'mean': class2_mean,
                'std': class2_std,
                'norm_greenness': class2_norm_greenness
            }
            
            # Create a visualization of the two classes
            visualization = np.zeros_like(img)
            # Class 1 - understory (shown in blue)
            visualization[class1_mask] = [255, 0, 0]
            # Class 2 - trees (shown in green) 
            visualization[class2_mask] = [0, 255, 0]
        
        return green_metrics, green_mask, class1_metrics, class2_metrics, visualization

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None, None

def quantify_vegetation_gmm(img):
    """
    Quantifies vegetation using Gaussian Mixture Models (GMM).
    
    GMMs can capture more complex cluster shapes than K-means and provide
    probability estimates for class membership. This implementation uses
    HSV color space for better vegetation segmentation.
    
    Args:
        img: The input image (BGR format)
        
    Returns:
        tuple: Overall green metrics (ratio, mean, std, norm_greenness),
               green mask, 
               class1 metrics (ratio, mean, std, norm_greenness), 
               class2 metrics (ratio, mean, std, norm_greenness), 
               class visualization
    """
    try:
        # First identify green pixels using GCC as before
        b, g, r = cv2.split(img)
        b = b.astype(float)
        g = g.astype(float)
        r = r.astype(float)
        
        greenness = g / (r + g + b + 1e-10)
        threshold = 0.37 # 0.37 +- 0.07
        green_mask = (greenness > threshold).astype(np.uint8) * 255
        
        green_pixels = np.sum(green_mask > 0)
        total_pixels = img.shape[0] * img.shape[1]
        green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0
        
        # Calculate mean and std greenness for all green pixels
        green_pixels_mask = green_mask > 0
        if green_pixels > 0:
            mean_greenness = np.mean(greenness[green_pixels_mask])
            std_greenness = np.std(greenness[green_pixels_mask])
            norm_greenness = np.sum(greenness[green_pixels_mask]) / green_pixels
        else:
            mean_greenness = 0
            std_greenness = 0
            norm_greenness = 0
            
        # Store overall green metrics as a dictionary
        green_metrics = {
            'ratio': green_ratio,
            'mean': mean_greenness,
            'std': std_greenness,
            'norm_greenness': norm_greenness
        }
        
        # Initialize class metrics
        class1_metrics = {'ratio': 0, 'mean': 0, 'std': 0, 'norm_greenness': 0}
        class2_metrics = {'ratio': 0, 'mean': 0, 'std': 0, 'norm_greenness': 0}
        visualization = np.zeros_like(img)
        
        # Use GMM for classification if we have enough green pixels
        non_zero_mask = green_mask > 0
        if np.sum(non_zero_mask) > 100:  # Require minimum number of green pixels
            # Convert to better color space for vegetation analysis
            masked_green = cv2.bitwise_and(img, img, mask=(green_mask // 255).astype(np.uint8))
            hsv_image = cv2.cvtColor(masked_green, cv2.COLOR_BGR2HSV)
            
            # Prepare data for GMM - extract features from pixels
            green_pixels_data = hsv_image[non_zero_mask].reshape(-1, 3)
            
            # Fit GMM with 2 components
            gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
            gmm.fit(green_pixels_data)
            
            # Predict classes
            labels = gmm.predict(green_pixels_data)
            
            # Create full masks
            full_labels = np.zeros(img.shape[:2], dtype=int)
            full_labels[non_zero_mask] = labels + 1  # +1 so background is 0
            
            # Determine which class is likely understory vs trees
            # Using means of H and V channels in HSV to distinguish
            means = gmm.means_
            if means[0][0] < means[1][0]:  # Lower hue value = more green-blue
                class1_mask = full_labels == 1  # Understory (darker)
                class2_mask = full_labels == 2  # Trees (brighter)
            else:
                class1_mask = full_labels == 2  # Understory (darker)
                class2_mask = full_labels == 1  # Trees (brighter)
                
            class1_pixels = np.sum(class1_mask)
            class2_pixels = np.sum(class2_mask)
            
            class1_ratio = class1_pixels / total_pixels if total_pixels > 0 else 0
            class2_ratio = class2_pixels / total_pixels if total_pixels > 0 else 0
            
            # Calculate mean, std, and normalized greenness for each class
            if class1_pixels > 0:
                class1_mean = np.mean(greenness[class1_mask])
                class1_std = np.std(greenness[class1_mask])
                class1_norm_greenness = np.sum(greenness[class1_mask]) / class1_pixels
            else:
                class1_mean = 0
                class1_std = 0
                class1_norm_greenness = 0
                
            if class2_pixels > 0:
                class2_mean = np.mean(greenness[class2_mask])
                class2_std = np.std(greenness[class2_mask])
                class2_norm_greenness = np.sum(greenness[class2_mask]) / class2_pixels
            else:
                class2_mean = 0
                class2_std = 0
                class2_norm_greenness = 0
                
            # Store class metrics
            class1_metrics = {
                'ratio': class1_ratio,
                'mean': class1_mean,
                'std': class1_std,
                'norm_greenness': class1_norm_greenness
            }
            
            class2_metrics = {
                'ratio': class2_ratio,
                'mean': class2_mean,
                'std': class2_std,
                'norm_greenness': class2_norm_greenness
            }
            
            # Visualization
            visualization = np.zeros_like(img)
            visualization[class1_mask] = [255, 0, 0]  # Understory in blue
            visualization[class2_mask] = [0, 255, 0]  # Trees in green
            
        return green_metrics, green_mask, class1_metrics, class2_metrics, visualization
        
    except Exception as e:
        print(f"An error occurred in GMM clustering: {e}")
        return None, None, None, None, None
    
def quantify_vegetation_dbscan(img):
    """
    Quantifies vegetation using DBSCAN clustering.
    
    DBSCAN is particularly good at finding arbitrarily shaped clusters and handling noise.
    This implementation combines both color and spatial features to account for both
    appearance and location of vegetation in the image.
    
    Args:
        img: The input image (BGR format)
        
    Returns:
        tuple: Overall green metrics (ratio, mean, std, norm_greenness),
               green mask, 
               class1 metrics (ratio, mean, std, norm_greenness), 
               class2 metrics (ratio, mean, std, norm_greenness), 
               class visualization
    """
    try:
        # Identify green pixels using GCC
        b, g, r = cv2.split(img)
        b, g, r = b.astype(float), g.astype(float), r.astype(float)
        greenness = g / (r + g + b + 1e-10)
        threshold = 0.37 # 0.37 +- 0.07
        green_mask = (greenness > threshold).astype(np.uint8) * 255
        
        green_pixels = np.sum(green_mask > 0)
        total_pixels = img.shape[0] * img.shape[1]
        green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0
        
        # Calculate mean and std greenness for all green pixels
        green_pixels_mask = green_mask > 0
        if green_pixels > 0:
            mean_greenness = np.mean(greenness[green_pixels_mask])
            std_greenness = np.std(greenness[green_pixels_mask])
            norm_greenness = np.sum(greenness[green_pixels_mask]) / green_pixels
        else:
            mean_greenness = 0
            std_greenness = 0
            norm_greenness = 0
            
        # Store overall green metrics as a dictionary
        green_metrics = {
            'ratio': green_ratio,
            'mean': mean_greenness,
            'std': std_greenness,
            'norm_greenness': norm_greenness
        }
        
        # Initialize class metrics
        class1_metrics = {'ratio': 0, 'mean': 0, 'std': 0, 'norm_greenness': 0}
        class2_metrics = {'ratio': 0, 'mean': 0, 'std': 0, 'norm_greenness': 0}
        visualization = np.zeros_like(img)
        
        # Apply DBSCAN if enough green pixels
        non_zero_mask = green_mask > 0
        if np.sum(non_zero_mask) > 100:
            # Get green pixel coordinates and values
            y_coords, x_coords = np.where(non_zero_mask)
            masked_green = cv2.bitwise_and(img, img, mask=(green_mask // 255).astype(np.uint8))
            lab_image = cv2.cvtColor(masked_green, cv2.COLOR_BGR2LAB)
            
            # Combine color and spatial features (with appropriate weighting)
            # Adjusting spatial weight increases/decreases importance of pixel position
            spatial_weight = 0.01  # Can be tuned
            features = np.column_stack([
                lab_image[non_zero_mask],          # Color features
                x_coords * spatial_weight,         # Spatial X coordinate
                y_coords * spatial_weight          # Spatial Y coordinate
            ])
            
            # Scale features to have unit variance
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Apply DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=10)
            labels = dbscan.fit_predict(scaled_features)
            
            # Get the two largest clusters (ignoring noise = -1)
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels >= 0]  # Remove noise
            
            if len(unique_labels) >= 2:
                # Count pixels in each cluster
                counts = np.array([np.sum(labels == label) for label in unique_labels])
                
                # Get the two largest clusters
                largest_indices = np.argsort(counts)[-2:]
                class1_label = unique_labels[largest_indices[1]]  # Largest cluster
                class2_label = unique_labels[largest_indices[0]]  # Second largest
                
                # Create full mask for visualization
                class1_indices = np.where(labels == class1_label)[0]
                class2_indices = np.where(labels == class2_label)[0]
                
                # Map back to original coordinates
                y1, x1 = y_coords[class1_indices], x_coords[class1_indices]
                y2, x2 = y_coords[class2_indices]
                
                # Create masks
                class1_mask = np.zeros(img.shape[:2], dtype=bool)
                class2_mask = np.zeros(img.shape[:2], dtype=bool)
                class1_mask[y1, x1] = True
                class2_mask[y2, x2] = True
                
                # Calculate ratios
                class1_pixels = np.sum(class1_mask)
                class2_pixels = np.sum(class2_mask)
                class1_ratio = class1_pixels / total_pixels if total_pixels > 0 else 0
                class2_ratio = class2_pixels / total_pixels if total_pixels > 0 else 0
                
                # Calculate mean, std, and normalized greenness for each class
                if class1_pixels > 0:
                    class1_mean = np.mean(greenness[class1_mask])
                    class1_std = np.std(greenness[class1_mask])
                    class1_norm_greenness = np.sum(greenness[class1_mask]) / class1_pixels
                else:
                    class1_mean = 0
                    class1_std = 0
                    class1_norm_greenness = 0
                    
                if class2_pixels > 0:
                    class2_mean = np.mean(greenness[class2_mask])
                    class2_std = np.std(greenness[class2_mask])
                    class2_norm_greenness = np.sum(greenness[class2_mask]) / class2_pixels
                else:
                    class2_mean = 0
                    class2_std = 0
                    class2_norm_greenness = 0
                    
                # Store class metrics
                class1_metrics = {
                    'ratio': class1_ratio,
                    'mean': class1_mean,
                    'std': class1_std,
                    'norm_greenness': class1_norm_greenness
                }
                
                class2_metrics = {
                    'ratio': class2_ratio,
                    'mean': class2_mean,
                    'std': class2_std,
                    'norm_greenness': class2_norm_greenness
                }
                
                # Create visualization
                visualization = np.zeros_like(img)
                visualization[class1_mask] = [255, 0, 0]  # Likely understory 
                visualization[class2_mask] = [0, 255, 0]  # Likely trees
            else:
                # Not enough clusters found - use all green pixels as class1
                class1_metrics = {
                    'ratio': green_ratio,
                    'mean': mean_greenness,
                    'std': std_greenness,
                    'norm_greenness': norm_greenness
                }
                
                visualization = np.zeros_like(img)
                visualization[non_zero_mask] = [255, 0, 0]
        
        return green_metrics, green_mask, class1_metrics, class2_metrics, visualization
        
    except Exception as e:
        print(f"An error occurred in DBSCAN clustering: {e}")
        return None, None, None, None, None

def quantify_vegetation_spectral(img):
    """
    Quantifies vegetation using spectral clustering.
    
    Spectral clustering is better at finding complex, non-linearly separable clusters.
    For large images, this implementation uses a sampling strategy to reduce computation
    time while maintaining good classification accuracy.
    
    Args:
        img: The input image (BGR format)
        
    Returns:
        tuple: Overall green metrics (ratio, mean, std, norm_greenness),
               green mask, 
               class1 metrics (ratio, mean, std, norm_greenness), 
               class2 metrics (ratio, mean, std, norm_greenness), 
               class visualization
    """
    try:
        # Identify green pixels
        b, g, r = cv2.split(img)
        b, g, r = b.astype(float), g.astype(float), r.astype(float)
        greenness = g / (r + g + b + 1e-10)
        threshold = 0.37 # 0.37 +- 0.07
        green_mask = (greenness > threshold).astype(np.uint8) * 255
        
        green_pixels = np.sum(green_mask > 0)
        total_pixels = img.shape[0] * img.shape[1]
        green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0
        
        # Calculate mean and std greenness for all green pixels
        green_pixels_mask = green_mask > 0
        if green_pixels > 0:
            mean_greenness = np.mean(greenness[green_pixels_mask])
            std_greenness = np.std(greenness[green_pixels_mask])
            norm_greenness = np.sum(greenness[green_pixels_mask]) / green_pixels
        else:
            mean_greenness = 0
            std_greenness = 0
            norm_greenness = 0
            
        # Store overall green metrics as a dictionary
        green_metrics = {
            'ratio': green_ratio,
            'mean': mean_greenness,
            'std': std_greenness,
            'norm_greenness': norm_greenness
        }
        
        # Initialize class metrics
        class1_metrics = {'ratio': 0, 'mean': 0, 'std': 0, 'norm_greenness': 0}
        class2_metrics = {'ratio': 0, 'mean': 0, 'std': 0, 'norm_greenness': 0}
        visualization = np.zeros_like(img)
        
        # Apply spectral clustering
        non_zero_mask = green_mask > 0
        if np.sum(non_zero_mask) > 100:
            # Extract green pixels in LAB color space
            masked_green = cv2.bitwise_and(img, img, mask=(green_mask // 255).astype(np.uint8))
            lab_image = cv2.cvtColor(masked_green, cv2.COLOR_BGR2LAB)
            
            # Get features for clustering
            features = lab_image[non_zero_mask].reshape(-1, 3)
            
            # Apply spectral clustering
            # Limit number of samples to avoid memory issues
            max_samples = 10000
            if len(features) > max_samples:
                # Random sampling of pixels
                indices = np.random.choice(len(features), max_samples, replace=False)
                sampled_features = features[indices]
                
                # Fit on samples
                spectral = SpectralClustering(n_clusters=2, random_state=42, 
                                             assign_labels='kmeans')
                sampled_labels = spectral.fit_predict(sampled_features)
                
                # Train a simple classifier on labeled samples for prediction
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(sampled_features, sampled_labels)
                
                # Predict on all features
                labels = knn.predict(features)
            else:
                # Direct clustering on all pixels
                spectral = SpectralClustering(n_clusters=2, random_state=42,
                                             assign_labels='kmeans')
                labels = spectral.fit_predict(features)
            
            # Create full masks
            class1_mask = np.zeros(img.shape[:2], dtype=bool)
            class2_mask = np.zeros(img.shape[:2], dtype=bool)
            
            y_indices, x_indices = np.where(non_zero_mask)
            class1_indices = np.where(labels == 0)[0]
            class2_indices = np.where(labels == 1)[0]
            
            # Map indices back to image coordinates
            class1_mask[y_indices[class1_indices], x_indices[class1_indices]] = True
            class2_mask[y_indices[class2_indices], x_indices[class2_indices]] = True
            
            # Calculate individual class ratios
            class1_pixels = np.sum(class1_mask)
            class2_pixels = np.sum(class2_mask)
            class1_ratio = class1_pixels / total_pixels if total_pixels > 0 else 0
            class2_ratio = class2_pixels / total_pixels if total_pixels > 0 else 0
            
            # Calculate mean, std, and normalized greenness for each class
            if class1_pixels > 0:
                class1_mean = np.mean(greenness[class1_mask])
                class1_std = np.std(greenness[class1_mask])
                class1_norm_greenness = np.sum(greenness[class1_mask]) / class1_pixels
            else:
                class1_mean = 0
                class1_std = 0
                class1_norm_greenness = 0
                
            if class2_pixels > 0:
                class2_mean = np.mean(greenness[class2_mask])
                class2_std = np.std(greenness[class2_mask])
                class2_norm_greenness = np.sum(greenness[class2_mask]) / class2_pixels
            else:
                class2_mean = 0
                class2_std = 0
                class2_norm_greenness = 0
                
            # Store class metrics
            class1_metrics = {
                'ratio': class1_ratio,
                'mean': class1_mean,
                'std': class1_std,
                'norm_greenness': class1_norm_greenness
            }
            
            class2_metrics = {
                'ratio': class2_ratio,
                'mean': class2_mean,
                'std': class2_std,
                'norm_greenness': class2_norm_greenness
            }
            
            # Create visualization
            visualization = np.zeros_like(img)
            visualization[class1_mask] = [255, 0, 0]  # Likely understory 
            visualization[class2_mask] = [0, 255, 0]  # Likely trees
            
        return green_metrics, green_mask, class1_metrics, class2_metrics, visualization
        
    except Exception as e:
        print(f"An error occurred in spectral clustering: {e}")
        return None, None, None, None, None
    
def quantify_vegetation(img, method="kmeans"):
    """
    Quantifies vegetation using various clustering methods.
    
    Args:
        img: The input image (BGR format)
        method: Clustering method to use ('kmeans', 'gmm', 'dbscan', 'spectral')
        
    Returns:
        tuple: Overall green metrics (ratio, mean, std, norm_greenness),
               green mask, 
               class1 metrics (ratio, mean, std, norm_greenness), 
               class2 metrics (ratio, mean, std, norm_greenness), 
               class visualization
    """
    if method == "kmeans":
        print("Using K-means clustering")
        return quantify_vegetation_kmeans(img)
    elif method == "gmm":
        print("Using Gaussian Mixture Models")
        return quantify_vegetation_gmm(img)
    elif method == "dbscan":
        print("Using DBSCAN clustering")
        return quantify_vegetation_dbscan(img)
    elif method == "spectral":
        print("Using Spectral clustering")
        return quantify_vegetation_spectral(img)
    else:
        print(f"Unknown method '{method}', using kmeans instead")
        return quantify_vegetation_kmeans(img)

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
    
    # Quantify vegetation within the whole image using the selected method
    green_metrics, green_mask, class1_metrics, class2_metrics, class_vis = quantify_vegetation(img, method=classification_method)

    if green_metrics is not None:
        print(f"The green pixel ratio is: {green_metrics['ratio']:.4f}")
        print(f"Green mean: {green_metrics['mean']:.4f}, std: {green_metrics['std']:.4f}, normalized: {green_metrics['norm_greenness']:.4f}")
        print(f"Class 1 ratio: {class1_metrics['ratio']:.4f}, mean: {class1_metrics['mean']:.4f}, std: {class1_metrics['std']:.4f}")
        print(f"Class 2 ratio: {class2_metrics['ratio']:.4f}, mean: {class2_metrics['mean']:.4f}, std: {class2_metrics['std']:.4f}")
        
        # Write result to CSV immediately
        with open(csv_path, 'a') as f:
            f.write(f'{i},{datetime_str},'
                    f'{green_metrics["ratio"]},{green_metrics["mean"]},{green_metrics["std"]},{green_metrics["norm_greenness"]},'
                    f'{class1_metrics["ratio"]},{class1_metrics["mean"]},{class1_metrics["std"]},{class1_metrics["norm_greenness"]},'
                    f'{class2_metrics["ratio"]},{class2_metrics["mean"]},{class2_metrics["std"]},{class2_metrics["norm_greenness"]},'
                    f'{classification_method}\n')

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
        axes[1].set_title(f"Green Masked (GR={green_metrics['ratio']:.2f}, Norm={green_metrics['norm_greenness']:.2f})")
        axes[1].axis('off')  # Hide the axes
        
        # Display the vegetation classes
        axes[2].imshow(class_vis_rgb)
        axes[2].set_title(f"Vegetation Classes ({classification_method})\n"
                  f"class1={class1_metrics['ratio']:.2f} (Norm={class1_metrics['norm_greenness']:.2f})\n"
                  f"class2={class2_metrics['ratio']:.2f} (Norm={class2_metrics['norm_greenness']:.2f})")
        axes[2].axis('off')
        
        # Create custom legend elements
        legend_elements = [
            Patch(facecolor='blue', edgecolor='black', label='Class 1'),
            Patch(facecolor='green', edgecolor='black', label='Class 2')
        ]
        
        # Add legend to the bottom of the third subplot
        axes[2].legend(handles=legend_elements, loc='lower center', 
                  bbox_to_anchor=(0.5, -0.3), frameon=True, 
                  facecolor='white', edgecolor='black')

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
            f.write(f'{i},{datetime_str},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,{classification_method}\n')

print(f"Done! Results saved to {csv_path}")
