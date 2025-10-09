#%%
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cmocean as cm
from datetime import datetime
import re

#%%
# Set paths
input_root = '/home/geofsn/data/timelapsethermal'  # Change to your folder containing subfolders with CSVs
output_root = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/1_Simon/1_Abisko/6_Tower_Data/Tower Thermal images/2_Extracted_Data_Shunan/2021/North-facing-preview'  # Change to your desired output folder

os.makedirs(output_root, exist_ok=True)
csv_outputpath = os.path.join(output_root, 'thermal_stats.csv')
with open(csv_outputpath, 'w') as f:
    f.write('filename,time,mean_temp,std_temp,min_temp,max_temp\n')

#%% Find all CSV files in subfolders
csv_files = glob.glob(os.path.join(input_root, '**', '*.csv'), recursive=True)

for csv_path in csv_files:

    imdatetime = os.path.basename(csv_path).strip('.csv')
    print(f'Processing {imdatetime}... ({csv_files.index(csv_path)+1}/{len(csv_files)}) | Current time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    # Read thermal image from CSV
    thermal_data = np.genfromtxt(csv_path, delimiter=' ', skip_header=7, filling_values=np.nan)
    # Read the width from the 2nd line of the CSV file (extract only the number)
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        match = re.search(r'(\d+)', lines[4])
        width = int(match.group(1)) if match else None
        print(f'Image width: {width}')
    # Plot image
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f'Thermal Image - {imdatetime}')
    ax.axis('off')
    # Display thermal image with cmocean thermal colormap, limit temperature to -10 to 20
    im = ax.imshow(thermal_data, cmap=cm.cm.thermal, vmin=-10, vmax=20)
    # Add colorbar to the bottom
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
    cbar.set_label('Temperature (°C)')
    
    # Prepare output path
    rel_path = os.path.relpath(csv_path, input_root)
    png_path = os.path.join(output_root, 'images', os.path.basename(rel_path).strip('.csv') + '.png')
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    
    # Save plot as PNG
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    # image stats
    mean_temp = np.nanmean(thermal_data)
    std_temp = np.nanstd(thermal_data)
    min_temp = np.nanmin(thermal_data)
    max_temp = np.nanmax(thermal_data)
    with open(csv_outputpath, 'a') as f:
        f.write(f'{os.path.basename(csv_path)},{imdatetime},{mean_temp:.2f},{std_temp:.2f},{min_temp:.2f},{max_temp:.2f}\n')

print(f"Processed {len(csv_files)} thermal images.")