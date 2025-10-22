"""
Thermal Image Vignetting Correction with Parallel Processing

This script applies vignetting correction to thermal images in parallel.
- Loads a vignetting correction layer (vignetCorr)
- Processes all .mat files starting with "W" in the input folder
- Applies correction: correctedData = tempData - vignetCorr
- Saves corrected images to output folder

Dependencies:
- numpy
- scipy.io (loadmat, savemat)
- concurrent.futures (for parallel processing)
- tqdm (optional, for progress bar)

Author: Shunan Feng (shf@ign.ku.dk)
"""

import os
import numpy as np
from scipy.io import loadmat, savemat
import concurrent.futures
from functools import partial
import multiprocessing

# Optional: for progress bar
try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False
    print("tqdm not available, progress bar disabled")

# --- Configuration ---
vignet_corr_file = "/home/geofsn/GitHub/project-Abisko/src/tower/vignetCorrLayerSN10600001_110x295_4dec.csv"
input_folder = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/Tower thermal images/preview/all/matimages"
output_folder = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/Tower thermal images/preview/all/vignetCorrectedImages"


# Number of parallel workers (None = CPU count - 1)
max_workers = None

# --- Load vignetting correction layer ---
print(f"Loading vignetting correction from: {vignet_corr_file}")
vignetCorr = np.loadtxt(vignet_corr_file, delimiter=",")
print(f"Vignetting correction shape: {vignetCorr.shape}")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def process_single_file(filename, input_folder, output_folder, vignetCorr):
    """
    Process a single thermal image file with vignetting correction.
    
    Parameters:
    -----------
    filename : str
        Name of the .mat file to process
    input_folder : str
        Path to input folder
    output_folder : str
        Path to output folder
    vignetCorr : ndarray
        Vignetting correction array
    
    Returns:
    --------
    dict : Processing result with status and filename
    """
    try:
        # Load thermal image
        filepath = os.path.join(input_folder, filename)
        data = loadmat(filepath)
        
        if 'thermal_image' not in data:
            return {
                'filename': filename,
                'status': 'failed',
                'error': 'No thermal_image variable found'
            }
        
        tempData = data['thermal_image']
        
        # Apply vignetting correction
        correctedData = tempData - vignetCorr
        
        # Save corrected image
        output_path = os.path.join(output_folder, filename)
        savemat(output_path, {"thermalImage": correctedData})
        
        return {
            'filename': filename,
            'status': 'success',
            'error': None
        }
        
    except Exception as e:
        return {
            'filename': filename,
            'status': 'failed',
            'error': str(e)
        }

def main():
    """Main function to process thermal images in parallel."""
    
    # Get list of files to process
    all_files = sorted([f for f in os.listdir(input_folder) 
                       if f.endswith(".mat") and f.startswith("W")]) # temporary solution to filter only west facing images
    
    
    if not all_files:
        print(f"No matching files found in {input_folder}")
        return
    
    print(f"Found {len(all_files)} files to process")
    
    # Determine number of workers
    if max_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    else:
        num_workers = max_workers
    
    print(f"Using {num_workers} worker processes")
    
    # Create partial function with fixed parameters
    process_func = partial(process_single_file,
                          input_folder=input_folder,
                          output_folder=output_folder,
                          vignetCorr=vignetCorr)
    
    # Process files in parallel
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        if USE_TQDM:
            # With progress bar
            futures = {executor.submit(process_func, f): f for f in all_files}
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures),
                             desc="Processing thermal images"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    filename = futures[future]
                    print(f"Error processing {filename}: {str(e)}")
                    results.append({
                        'filename': filename,
                        'status': 'failed',
                        'error': str(e)
                    })
        else:
            # Without progress bar
            for result in executor.map(process_func, all_files):
                results.append(result)
                if result['status'] == 'failed':
                    print(f"Failed: {result['filename']} - {result['error']}")
    
    # Print summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    fail_count = sum(1 for r in results if r['status'] == 'failed')
    
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    print(f"Total files: {len(all_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Output folder: {output_folder}")
    
    if fail_count > 0:
        print("\nFailed files:")
        for r in results:
            if r['status'] == 'failed':
                print(f"  - {r['filename']}: {r['error']}")

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")