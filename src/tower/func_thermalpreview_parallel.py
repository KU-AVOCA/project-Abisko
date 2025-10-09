import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import cmocean as cm
import concurrent.futures
import pandas as pd
from functools import partial
import tqdm

def process_single_thermal_image(csv_path, input_root, output_root):
    """
    Process a single thermal CSV file and generate preview image with statistics.
    
    Parameters:
    csv_path (str): Path to the CSV file
    input_root (str): Root input directory
    output_root (str): Root output directory
    
    Returns:
    dict: Statistics for the processed image
    """
    imdatetime = os.path.basename(csv_path).strip('.csv')
    
    try:
        # Read thermal image from CSV
        thermal_data = np.genfromtxt(csv_path, delimiter=' ', skip_header=7, filling_values=np.nan)
        # Read the width from the 3rd line of the CSV file (extract only the number)
        # Read the height from the 4th line of the CSV file (extract only the number)
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            match = re.search(r'(\d+)', lines[3])
            imwidth = int(match.group(1)) if match else None
            # print(f'Image width: {imwidth}')
            match = re.search(r'(\d+)', lines[4])
            imheight = int(match.group(1)) if match else None
            # print(f'Image height: {imheight}')

        # if width 295 height 111 then it's west-facing
        # if width 300 height 120 then it's north-facing
        if imwidth == 295 and imheight == 111:
            imposition = 'West-facing'
            # print("Detected West-facing camera image")
        elif imwidth == 300 and imheight == 120:
            imposition = 'North-facing'
            # print("Detected North-facing camera image")
        else:
            imposition = 'Unknown'
            # print("Warning: Unrecognized image dimensions")

        # Plot image
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f'Thermal Image - {imdatetime}')
        ax.axis('off')
        # Display thermal image with cmocean thermal colormap
        im = ax.imshow(thermal_data, cmap=cm.cm.thermal)
        # Add colorbar to the bottom
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label('Temperature (°C)')
        
        # Prepare output path
        rel_path = os.path.relpath(csv_path, input_root)
        png_path = os.path.join(output_root, 'images', imposition + '_' + os.path.basename(rel_path).strip('.csv') + '.png')
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        npy_path = os.path.join(output_root, 'npyimages', imposition + '_' + os.path.basename(rel_path).strip('.csv') + '.npy')
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        
        # Save plot as PNG
        fig.savefig(png_path, bbox_inches='tight', dpi=300)
        plt.close('all')  # Close the figure to free memory
        # Save figure as numpy array
        np.save(npy_path, thermal_data)

        # Calculate image stats
        mean_temp = np.nanmean(thermal_data)
        std_temp = np.nanstd(thermal_data)
        min_temp = np.nanmin(thermal_data)
        max_temp = np.nanmax(thermal_data)
        
        # Return stats for CSV
        return {
            'filename': os.path.basename(csv_path),
            'time': imdatetime,
            'mean_temp': mean_temp,
            'std_temp': std_temp,
            'min_temp': min_temp,
            'max_temp': max_temp,
            'status': 'success',
            'imposition': imposition
        }
    
    except Exception as e:
        return {
            'filename': os.path.basename(csv_path),
            'time': imdatetime,
            'mean_temp': np.nan,
            'std_temp': np.nan,
            'min_temp': np.nan,
            'max_temp': np.nan,
            'status': 'failed',
            'error': str(e),
            'imposition': 'Unknown'
        }

def process_thermal_images_parallel(input_root, output_root, max_workers=None):
    """
    Process thermal CSV files in parallel and generate preview images with statistics.
    
    Parameters:
    input_root (str): Path to folder containing subfolders with CSV files
    output_root (str): Path to desired output folder
    max_workers (int): Maximum number of worker processes (default: CPU count - 1)
    """
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(os.path.join(output_root, 'images'), exist_ok=True)
    
    # Find all CSV files in subfolders
    csv_files = glob.glob(os.path.join(input_root, '**', '*.csv'), recursive=True)
    
    if not csv_files:
        print(f"No CSV files found in {input_root}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")

    # Determine optimal number of workers
    if max_workers is None:
        max_workers = os.cpu_count() - 1  # Leave one CPU free for system operations
        max_workers = max(1, min(max_workers, 8))  # Cap at 8 workers to avoid memory issues
    
    print(f"Using {max_workers} worker processes")
    
    # Create a partial function with the fixed parameters
    process_func = partial(process_single_thermal_image, 
                          input_root=input_root, 
                          output_root=output_root)
    
    # Process images in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a progress bar
        futures = {executor.submit(process_func, csv_path): csv_path for csv_path in csv_files}
        
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), 
                               total=len(futures), 
                               desc="Processing thermal images"):
            csv_path = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing {csv_path}: {str(e)}")
                results.append({
                    'filename': os.path.basename(csv_path),
                    'time': os.path.basename(csv_path).strip('.csv'),
                    'mean_temp': np.nan,
                    'std_temp': np.nan,
                    'min_temp': np.nan,
                    'max_temp': np.nan,
                    'status': 'failed',
                    'error': str(e),
                    'imposition': 'Unknown'
                })
    
    # Create DataFrame from results and save to CSV
    df = pd.DataFrame(results)
    csv_outputpath = os.path.join(output_root, 'thermal_stats.csv')
    df.to_csv(csv_outputpath, 
              columns=['filename', 'time', 'mean_temp', 'std_temp', 'min_temp', 'max_temp', 'imposition'], 
              index=False)
    
    # Print summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    fail_count = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"Processed {len(csv_files)} thermal images: {success_count} successful, {fail_count} failed")
    print(f"Images saved to: {os.path.join(output_root, 'images')}")
    print(f"Statistics saved to: {csv_outputpath}")

def main():
    """Main function to get user input and run the processing."""
    print("Thermal Image Preview Generator (Parallel)")
    print("=" * 45)
    
    # Get input root path
    while True:
        input_root = input("Enter the input root path (containing thermal images in CSV format): ").strip()
        if os.path.exists(input_root):
            break
        else:
            print(f"Error: Path '{input_root}' does not exist. Please try again.")
    
    # Get output root path
    output_root = input("Enter the output root path (for preview images): ").strip()
    
    # Get number of workers (optional)
    workers_input = input(f"Enter number of worker processes (default: {os.cpu_count()-1}): ").strip()
    max_workers = None
    if workers_input:
        try:
            max_workers = int(workers_input)
            max_workers = max(1, max_workers)  # Ensure at least 1 worker
        except ValueError:
            print("Invalid input for workers, using default.")
    
    # Confirm paths
    print(f"\nInput path: {input_root}")
    print(f"Output path: {output_root}")
    print(f"Max workers: {max_workers if max_workers else os.cpu_count()-1}")
    confirm = input("Proceed with these settings? (y/n): ").strip().lower()
    
    if confirm in ['y', 'yes']:
        process_thermal_images_parallel(input_root, output_root, max_workers)
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()