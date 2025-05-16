"""
CCDC Defoliation Analysis Script
This script processes daily Green Chromatic Coordinate (GCC) data using Continuous Change Detection 
and Classification (CCDC) to analyze vegetation defoliation patterns over multiple years.
The script:
1. Loads daily GCC TIFF files from a specified directory
2. Filters data for summer months (June, July, August)
3. Applies a water mask to exclude non-vegetated areas
4. Extracts Condition Score (CS) and Relative Deviation (RD) data from each TIFF
5. Calculates annual means of CS and RD for each pixel
6. Saves the annual means as numpy arrays for further analysis
The output consists of two numpy arrays:
- rd_annual.npy: Annual means of Relative Deviation (measure of deviation from expected GCC)
- cs_annual.npy: Annual means of Condition Score (measure of vegetation condition)
These outputs can be used to analyze inter-annual patterns in vegetation health and defoliation events.

Shunan Feng (shf@ign.ku.dk)
"""
#%%

import pandas as pd
import numpy as np
import glob
import os
# import matplotlib.pyplot as plt
# import contextily as ctx
import rasterio as rio
# from rasterio.plot import show
# import cmocean
# import cmcrameri
# from matplotlib.gridspec import GridSpec
# import seaborn as sns
# sns.set_theme(style="darkgrid", font_scale=1.5)
# %%
# Define folder containing TIFF files
imfolder = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/CCDC/data/GCCdaily/"
imoutfolder = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/CCDC/data/"
# Create output folder if it doesn't exist
if not os.path.exists(imoutfolder):
    os.makedirs(imoutfolder)
imfiles = sorted(glob.glob(os.path.join(imfolder, "*.tif")))
maskfile = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/CCDC/data/waterMask.tif"
with rio.open(maskfile) as src:
    watermask = src.read(1)
    improfile = src.profile
# Create a DataFrame to store file paths and metadata
imfiles = pd.DataFrame(imfiles, columns=["impath"])
imfiles["imname"] = imfiles.impath.str.split("/").str[-1]
imfiles["date"] = pd.to_datetime(
    imfiles.imname.str.split("_").str[2].str.replace(".tif", ""), format="%Y-%m-%d"
)

# Filter for June, July, and August only
imfiles["year"] = imfiles["date"].dt.year
imfiles["month"] = imfiles["date"].dt.month
imfiles = imfiles[imfiles["month"].isin([6, 7, 8])]


# %%
years = imfiles["year"].unique()
rd_annual = np.full((len(years), *watermask.shape), np.nan)
cs_annual = np.full((len(years), *watermask.shape), np.nan)

# Loop through each year and calculate the mean for each year
for i in range(len(years)):
    year = years[i]
    print(f"Processing year {year}...")
    imfiles_year = imfiles[imfiles["year"] == year].sort_values("date")
    rdCollection = np.full((len(imfiles_year), *watermask.shape), np.nan)
    csCollection = np.full((len(imfiles_year), *watermask.shape), np.nan)

    for j in range(len(imfiles_year)):
        imfile = imfiles_year.iloc[j]
        print(f"Processing {imfile['imname']}...")
        try:
            # Open the raster file using xarray with the rasterio engine
            with rio.open(imfile["impath"]) as src:

                conditionScore = src.read(4)
                conditionScore[watermask == 0] = np.nan
                relativeDeviation = src.read(5)
                relativeDeviation[watermask == 0] = np.nan
                relativeDeviation = np.where(
                    (relativeDeviation > 1) | (relativeDeviation < -1),
                    np.nan,
                    relativeDeviation
                )
                
                # Store the data in the collections
                rdCollection[j,:,:] = relativeDeviation
                csCollection[j,:,:] = conditionScore
        except Exception as e:
            print(f"Error processing {imfile['impath']}: {e}")
            continue
    # Calculate the mean across the time dimension for each year
    rd_annual[i,:,:] = np.nanmean(rdCollection, axis=0)
    cs_annual[i,:,:] = np.nanmean(csCollection, axis=0)

# Save the output as numpy arrays
np.save(os.path.join(imoutfolder, "rd_annual.npy"), rd_annual)
np.save(os.path.join(imoutfolder, "cs_annual.npy"), cs_annual)




# %%
