import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import cmocean as cm
from datetime import datetime

def process_thermal_images(input_root, output_root):
    """
    Process thermal CSV files and generate preview images with statistics.
    
    Parameters:
    input_root (str): Path to folder containing subfolders with CSV files
    output_root (str): Path to desired output folder
    """
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    csv_outputpath = os.path.join(output_root, 'thermal_stats.csv')
    
    # Initialize statistics CSV file
    with open(csv_outputpath, 'w') as f:
        f.write('filename,time,mean_temp,std_temp,min_temp,max_temp,imposition\n')

    # Find all CSV files in subfolders
    csv_files = glob.glob(os.path.join(input_root, '**', '*.csv'), recursive=True)
    
    if not csv_files:
        print(f"No CSV files found in {input_root}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")

    for csv_path in csv_files:
        imdatetime = os.path.basename(csv_path).strip('.csv')
        print(f'Processing {imdatetime}... ({csv_files.index(csv_path)+1}/{len(csv_files)}) | Current time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        imwidth = None
        imheight = None
        imposition = None

        try:
            # Read thermal image from CSV
            thermal_data = np.genfromtxt(csv_path, delimiter=' ', skip_header=7, filling_values=np.nan)
            # Read the width from the 4th line of the CSV file (extract only the number)
            # Read the height from the 5th line of the CSV file (extract only the number)
            with open(csv_path, 'r') as f:
                lines = f.readlines()
                match = re.search(r'(\d+)', lines[4])
                imwidth = int(match.group(1)) if match else None
                print(f'Image width: {imwidth}')
                match = re.search(r'(\d+)', lines[5])
                imheight = int(match.group(1)) if match else None
                print(f'Image height: {imheight}')

            # if width 295 height 111 then it's west-facing
            # if width 300 height 120 then it's north-facing
            if imwidth == 295 and imheight == 111:
                imposition = 'West-facing'
                print("Detected West-facing camera image")
            elif imwidth == 300 and imheight == 120:
                imposition = 'North-facing'
                print("Detected North-facing camera image")
            else:
                imposition = 'Unknown'
                print("Warning: Unrecognized image dimensions")

            # Plot image
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title(f'Thermal Image - {imdatetime}')
            ax.axis('off')
            # Display thermal image with cmocean thermal colormap, limit temperature to -10 to 20
            im = ax.imshow(thermal_data, cmap=cm.cm.thermal) #, vmin=-10, vmax=20
            # Add colorbar to the bottom
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
            cbar.set_label('Temperature (°C)')
            
            # Prepare output path
            rel_path = os.path.relpath(csv_path, input_root)
            png_path = os.path.join(output_root, 'images', os.path.basename(rel_path).strip('.csv') + imposition + '.png')
            os.makedirs(os.path.dirname(png_path), exist_ok=True)
            
            # Save plot as PNG
            fig.savefig(png_path, bbox_inches='tight', dpi=300)
            plt.close('all')

            # Calculate image stats
            mean_temp = np.nanmean(thermal_data)
            std_temp = np.nanstd(thermal_data)
            min_temp = np.nanmin(thermal_data)
            max_temp = np.nanmax(thermal_data)
            
            # Write stats to CSV
            with open(csv_outputpath, 'a') as f:
                f.write(f'{os.path.basename(csv_path)},{imdatetime},{mean_temp:.2f},{std_temp:.2f},{min_temp:.2f},{max_temp:.2f},{imposition}\n')
        
        except Exception as e:
            print(f"Error processing {csv_path}: {str(e)}")
            continue

    print(f"Processed {len(csv_files)} thermal images.")
    print(f"Images saved to: {os.path.join(output_root, 'images')}")
    print(f"Statistics saved to: {csv_outputpath}")

def main():
    """Main function to get user input and run the processing."""
    print("Thermal Image Preview Generator")
    print("=" * 40)
    
    # Get input root path
    while True:
        input_root = input("Enter the input root path (containing thermal images in CSV format): ").strip()
        if os.path.exists(input_root):
            break
        else:
            print(f"Error: Path '{input_root}' does not exist. Please try again.")
    
    # Get output root path
    output_root = input("Enter the output root path (for preview images): ").strip()
    
    # Confirm paths
    print(f"\nInput path: {input_root}")
    print(f"Output path: {output_root}")
    confirm = input("Proceed with these paths? (y/n): ").strip().lower()
    
    if confirm in ['y', 'yes']:
        process_thermal_images(input_root, output_root)
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()