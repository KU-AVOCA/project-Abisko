"""
Interactive tool for training a supervised vegetation classifier.
This script:
1. Filters for daytime images (based on solar elevation using pvlib)
2. Randomly selects 100 images for training
3. Allows manual pixel selection for birch, understory, and non-green classes
4. Trains a Random Forest classifier
5. Saves the trained model for later use

Author: Shunan Feng (shf@ign.ku.dk)
"""
#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS
import datetime
from tqdm import tqdm
import pvlib

# Define Abisko coordinates
SITE_LATITUDE = 68.34808742  # in decimal degrees
SITE_LONGITUDE = 19.05077561  # in decimal degrees
SITE_ELEVATION = 400  # meters above sea level (approximate)

class PixelSelector:
    def __init__(self):
        self.birch_pixels = []
        self.understory_pixels = []
        self.nongreen_pixels = []
        self.current_class = 'birch'
        self.image = None
        self.lab_image = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Callback for mouse clicks to select pixels"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get LAB values at clicked position
            lab_value = self.lab_image[y, x]
            rgb_value = self.image[y, x]
            
            pixel_info = {
                'lab': lab_value,
                'rgb': rgb_value,
                'position': (x, y)
            }
            
            if self.current_class == 'birch':
                self.birch_pixels.append(pixel_info)
                cv2.circle(self.display_image, (x, y), 3, (0, 255, 0), -1)
                print(f"Birch pixel {len(self.birch_pixels)}: LAB={lab_value}")
            elif self.current_class == 'understory':
                self.understory_pixels.append(pixel_info)
                cv2.circle(self.display_image, (x, y), 3, (255, 0, 0), -1)
                print(f"Understory pixel {len(self.understory_pixels)}: LAB={lab_value}")
            elif self.current_class == 'nongreen':
                self.nongreen_pixels.append(pixel_info)
                cv2.circle(self.display_image, (x, y), 3, (0, 0, 255), -1)
                print(f"Non-green pixel {len(self.nongreen_pixels)}: LAB={lab_value}")
            
            cv2.imshow('Image', self.display_image)
    
    def select_pixels_from_image(self, image_path):
        """Interactive pixel selection from a single image"""
        self.image = cv2.imread(image_path)
        if self.image is None:
            print(f"Could not read image: {image_path}")
            return False
        
        # Apply same cropping as main script
        crop_top = 800
        buffer = 50
        self.image = self.image[crop_top:, :, :]
        self.image = self.image[buffer:-buffer, buffer:-buffer, :]
        
        # Convert to LAB
        self.lab_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        self.display_image = self.image.copy()
        
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', self.mouse_callback)
        
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        print("Instructions:")
        print("  'b' - Switch to Birch selection (green markers)")
        print("  'u' - Switch to Understory selection (blue markers)")
        print("  'n' - Switch to Non-green selection (red markers)")
        print("  's' - Skip to next image")
        print("  'q' - Quit and save")
        print(f"\nCurrent class: {self.current_class.upper()}")
        print(f"Counts - Birch: {len(self.birch_pixels)}, "
              f"Understory: {len(self.understory_pixels)}, "
              f"Non-green: {len(self.nongreen_pixels)}")
        
        cv2.imshow('Image', self.display_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('b'):
                self.current_class = 'birch'
                print("\n>>> Switched to BIRCH selection (green markers)")
            elif key == ord('u'):
                self.current_class = 'understory'
                print("\n>>> Switched to UNDERSTORY selection (blue markers)")
            elif key == ord('n'):
                self.current_class = 'nongreen'
                print("\n>>> Switched to NON-GREEN selection (red markers)")
            elif key == ord('s'):
                print("\n>>> Skipping to next image...")
                break
            elif key == ord('q'):
                print("\n>>> Quitting pixel selection...")
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()
        return True

def get_image_rgb_datetime(image_path):
    """Extract the datetime when the image was taken from EXIF metadata."""
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
                    return dt
                except ValueError:
                    return None
                    
            return None
    except Exception as e:
        print(f"Error extracting metadata from {image_path}: {e}")
        return None

def is_daytime(timestamp, min_elevation=5.0):
    """
    Determine if a timestamp is during daylight hours based on solar elevation.
    
    Args:
        timestamp: datetime object
        min_elevation: Minimum solar elevation angle (in degrees) to be considered daytime
                      (5 degrees excludes dawn/dusk periods)
    
    Returns:
        bool: True if the timestamp is during daylight hours
    """
    try:
        if timestamp is None or pd.isna(timestamp):
            return False
        
        # Calculate solar position
        solpos = pvlib.solarposition.get_solarposition(
            timestamp, 
            SITE_LATITUDE, 
            SITE_LONGITUDE,
            altitude=SITE_ELEVATION
        )
        
        # Get solar elevation angle
        elevation = solpos['elevation'].iloc[0]
        
        # Check if it's daytime (sun is above the minimum elevation)
        return elevation > min_elevation
        
    except Exception as e:
        print(f"Error calculating solar position for {timestamp}: {e}")
        return False

def filter_daytime_images(image_files, min_elevation=5.0):
    """
    Filter images to only include daytime images based on solar elevation using pvlib.
    
    Args:
        image_files: List of image file paths
        min_elevation: Minimum solar elevation angle (in degrees) to be considered daytime
        
    Returns:
        List of daytime image file paths
    """
    daytime_images = []
    
    print(f"\nFiltering for daytime images (solar elevation > {min_elevation}°)...")
    for img_path in tqdm(image_files, desc="Checking image times"):
        dt = get_image_rgb_datetime(img_path)
        if dt is not None and is_daytime(dt, min_elevation=min_elevation):
            daytime_images.append(img_path)
    
    print(f"Found {len(daytime_images)} daytime images out of {len(image_files)} total images")
    return daytime_images

def collect_training_data(image_files, n_images=100, output_file='training_data.pkl'):
    """
    Collect training pixels from multiple images.
    
    Args:
        image_files: List of image file paths
        n_images: Number of images to use for training
        output_file: Path to save the training data
        
    Returns:
        Dictionary containing training pixel data
    """
    selector = PixelSelector()
    
    # Randomly select n_images
    n_to_select = min(n_images, len(image_files))
    selected_images = np.random.choice(image_files, n_to_select, replace=False)
    
    print(f"\n{'='*60}")
    print(f"Starting pixel selection from {n_to_select} images")
    print(f"{'='*60}\n")
    
    for idx, img_path in enumerate(selected_images, 1):
        print(f"\nImage {idx}/{n_to_select}")
        if not selector.select_pixels_from_image(img_path):
            break
    
    # Save training data
    training_data = {
        'birch': selector.birch_pixels,
        'understory': selector.understory_pixels,
        'nongreen': selector.nongreen_pixels
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(training_data, f)
    
    print(f"\n{'='*60}")
    print(f"Training data saved to {output_file}")
    print(f"{'='*60}")
    print(f"Total pixels collected:")
    print(f"  Birch:       {len(selector.birch_pixels)}")
    print(f"  Understory:  {len(selector.understory_pixels)}")
    print(f"  Non-green:   {len(selector.nongreen_pixels)}")
    print(f"  TOTAL:       {len(selector.birch_pixels) + len(selector.understory_pixels) + len(selector.nongreen_pixels)}")
    
    return training_data

def train_classifier(training_data_file='training_data.pkl', output_model='vegetation_classifier.pkl'):
    """
    Train a Random Forest classifier from collected training data.
    
    Args:
        training_data_file: Path to the training data pickle file
        output_model: Path to save the trained model
        
    Returns:
        Trained classifier
    """
    print(f"\n{'='*60}")
    print("Training Random Forest Classifier")
    print(f"{'='*60}\n")
    
    # Load training data
    with open(training_data_file, 'rb') as f:
        training_data = pickle.load(f)
    
    # Prepare features and labels
    X = []
    y = []
    
    for pixel in training_data['birch']:
        X.append(pixel['lab'])
        y.append(0)  # Birch = 0
    
    for pixel in training_data['understory']:
        X.append(pixel['lab'])
        y.append(1)  # Understory = 1
    
    for pixel in training_data['nongreen']:
        X.append(pixel['lab'])
        y.append(2)  # Non-green = 2
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Training classifier with {len(X)} samples")
    print(f"Class distribution:")
    print(f"  Birch (0):      {np.sum(y==0)} samples")
    print(f"  Understory (1): {np.sum(y==1)} samples")
    print(f"  Non-green (2):  {np.sum(y==2)} samples")
    
    # Check for class imbalance
    min_samples = min(np.sum(y==0), np.sum(y==1), np.sum(y==2))
    if min_samples < 10:
        print(f"\nWARNING: Very few samples for at least one class (min={min_samples})")
        print("Consider collecting more training data for better results.")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set:     {len(X_test)} samples")
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    
    print(f"\nTraining accuracy: {train_score:.4f}")
    print(f"Test accuracy:     {test_score:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Birch', 'Understory', 'Non-green']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Birch', 'Understory', 'Non-green'],
                yticklabels=['Birch', 'Understory', 'Non-green'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("\nConfusion matrix saved to confusion_matrix.png")
    
    # Feature importance
    plt.figure(figsize=(8, 6))
    feature_names = ['L (Lightness)', 'a (green-red)', 'b (blue-yellow)']
    importances = clf.feature_importances_
    plt.bar(feature_names, importances)
    plt.title('Feature Importance')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    print("Feature importance plot saved to feature_importance.png")
    
    # Save model
    model_data = {
        'classifier': clf,
        'training_accuracy': train_score,
        'test_accuracy': test_score,
        'n_samples': len(X),
        'class_distribution': {
            'birch': np.sum(y==0),
            'understory': np.sum(y==1),
            'nongreen': np.sum(y==2)
        }
    }
    
    with open(output_model, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n{'='*60}")
    print(f"Model saved to {output_model}")
    print(f"{'='*60}\n")
    
    return clf

if __name__ == "__main__":
    # Set paths
    rgbfolder = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/1_Simon/1_Abisko/6_Tower_Data/Tower RGB images/1 Data/1 Years'
    
    print(f"\n{'='*60}")
    print("Supervised Vegetation Classifier Training")
    print(f"Using pvlib for solar position calculation")
    print(f"Site: Abisko ({SITE_LATITUDE}°N, {SITE_LONGITUDE}°E, {SITE_ELEVATION}m)")
    print(f"{'='*60}\n")
    
    # Find all RGB images
    print("Searching for RGB images...")
    imrgbfiles = []
    imrgbfiles.extend(glob.glob(os.path.join(rgbfolder, '**/', '*.JPG'), recursive=True))
    imrgbfiles.extend(glob.glob(os.path.join(rgbfolder, '**/', '*.jpg'), recursive=True))
    imrgbfiles.extend(glob.glob(os.path.join(rgbfolder, '**/', '*.JPEG'), recursive=True))
    imrgbfiles.extend(glob.glob(os.path.join(rgbfolder, '**/', '*.jpeg'), recursive=True))
    imrgbfiles.extend(glob.glob(os.path.join(rgbfolder, '**/', '*.png'), recursive=True))
    imrgbfiles.extend(glob.glob(os.path.join(rgbfolder, '**/', '*.PNG'), recursive=True))
    print(f"Found {len(imrgbfiles)} total images")
    
    # Filter for West-facing only
    print("\nFiltering for West-facing images...")
    imrgbfiles = [f for f in imrgbfiles if 'West' in f]
    print(f"Found {len(imrgbfiles)} West-facing images")
    
    # Filter for daytime images using solar elevation
    daytime_images = filter_daytime_images(imrgbfiles, min_elevation=5.0)
    
    if len(daytime_images) == 0:
        print("\nERROR: No daytime images found!")
        print("Check if EXIF data is available in your images.")
        exit(1)
    
    # Step 1: Collect training data
    print(f"\n{'='*60}")
    print("Step 1: Collecting Training Data")
    print(f"{'='*60}")
    training_data = collect_training_data(
        daytime_images,
        n_images=100,
        output_file='training_data.pkl'
    )
    
    # Check if enough data was collected
    total_pixels = (len(training_data['birch']) + 
                   len(training_data['understory']) + 
                   len(training_data['nongreen']))
    
    if total_pixels < 30:
        print("\nWARNING: Very few training samples collected!")
        print("Consider running the script again and collecting more pixels.")
    
    # Step 2: Train classifier
    print(f"\n{'='*60}")
    print("Step 2: Training Classifier")
    print(f"{'='*60}")
    clf = train_classifier('training_data.pkl', 'vegetation_classifier.pkl')
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print("\nFiles created:")
    print("  - training_data.pkl          : Training pixel data")
    print("  - vegetation_classifier.pkl  : Trained model")
    print("  - confusion_matrix.png       : Model evaluation")
    print("  - feature_importance.png     : Feature importance plot")
    print("\nTo use the trained model:")
    print("  Set classification_method = 'supervised' in your main script")
    print(f"{'='*60}\n")
# %%
