#%%
import pandas as pd
import numpy as np
import xarray as xr
import glob
import os
import matplotlib.pyplot as plt
import contextily as ctx
import rasterio as rio
from rasterio.plot import show
import cmocean
import cmcrameri
from matplotlib.gridspec import GridSpec
import seaborn as sns
sns.set_theme(style="darkgrid", font_scale=1.5)
# %%
# Define folder containing TIFF files
imfolder = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/CCDC/data/GCCdaily/"
imoutfolder = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/CCDC/data/GCCdaily_preview/"
# Create output folder if it doesn't exist
if not os.path.exists(imoutfolder):
    os.makedirs(imoutfolder)
imfiles = sorted(glob.glob(os.path.join(imfolder, "*.tif")))
maskfile = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/CCDC/data/waterMask.tif"
with rio.open(maskfile) as src:
    watermask = src.read(1)
# Create a DataFrame to store file paths and metadata
imfiles = pd.DataFrame(imfiles, columns=["impath"])
imfiles["imname"] = imfiles.impath.str.split("/").str[-1]
imfiles["date"] = pd.to_datetime(
    imfiles.imname.str.split("_").str[2].str.replace(".tif", ""), format="%Y-%m-%d"
)

#%% optional: open and plot GCC, GCC_predicted, conditionScore, and relativeDeviation
# // band1: GCC_predicted
# // band2: GCC_RMSE
# // band3: GCC
# // band4: conditionScore
# // band5: relativeDeviation
for i in range(0, len(imfiles)):
    try:
        # Open the raster file using xarray with the rasterio engine
        with rio.open(imfiles["impath"].values[i]) as src:
            GCC_predicted = src.read(1)
            GCC_predicted[watermask == 0] = np.nan
            GCC = src.read(3)
            GCC[watermask == 0] = np.nan
            conditionScore = src.read(4)
            conditionScore[watermask == 0] = np.nan
            relativeDeviation = src.read(5)
            relativeDeviation[watermask == 0] = np.nan
            
            # Create a figure with subplots for 2x2 layout
            fig = plt.figure(figsize=(16, 8))
            gs = GridSpec(2, 2, height_ratios=[1, 1])
            ax1 = fig.add_subplot(gs[0, 0])
            show(
                GCC,
                transform=src.transform,
                ax=ax1,
                cmap=cmocean.cm.algae,
                vmin=0,
                vmax=1
            )
            ctx.add_basemap(
                ax1,
                crs=src.crs,
                attribution=f"© OpenStreetMap contributors, CRS: {src.crs.to_string()}"
            )
            show(
                GCC,
                transform=src.transform,
                ax=ax1,
                cmap=cmocean.cm.algae,
                vmin=0,
                vmax=1
            )            
            cbar1 = plt.colorbar(
                plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=1), cmap=cmocean.cm.algae),
                ax=ax1,
                shrink=0.8
            )
            cbar1.set_label("GCC")
            ax2 = fig.add_subplot(gs[0, 1])
            show(
                GCC_predicted,
                transform=src.transform,
                ax=ax2,
                cmap=cmocean.cm.algae,
                vmin=0,
                vmax=1
            )
            ctx.add_basemap(
                ax2,
                crs=src.crs,
                attribution=f"© OpenStreetMap contributors, CRS: {src.crs.to_string()}"
            )
            show(
                GCC_predicted,
                transform=src.transform,
                ax=ax2,
                cmap=cmocean.cm.algae,
                vmin=0,
                vmax=1
            )
            cbar2 = plt.colorbar(
                plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=1), cmap=cmocean.cm.algae),
                ax=ax2,
                shrink=0.8
            )
            cbar2.set_label("GCC_predicted")
            ax3 = fig.add_subplot(gs[1, 0])
            show(
                conditionScore,
                transform=src.transform,
                ax=ax3,
                cmap=cmocean.cm.balance_r,
                # cmap=cmcrameri.cm.hawaii,
                vmin=-4,
                vmax=4
            )
            ctx.add_basemap(
                ax3,
                crs=src.crs,
                attribution=f"© OpenStreetMap contributors, CRS: {src.crs.to_string()}"
            )
            show(
                conditionScore,
                transform=src.transform,
                ax=ax3,
                cmap=cmocean.cm.balance_r,
                # cmap=cmcrameri.cm.hawaii,
                vmin=-4,
                vmax=4
            )
            cbar3 = plt.colorbar(
                plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-4, vmax=4), cmap=cmocean.cm.balance_r),
                ax=ax3,
                shrink=0.8,
                extend="both"
            )
            cbar3.set_label("condition score")
            ax4 = fig.add_subplot(gs[1, 1])
            show(
                relativeDeviation,
                transform=src.transform,
                ax=ax4,
                cmap=cmocean.cm.balance,
                vmin=-1,
                vmax=1
            )
            ctx.add_basemap(
                ax4,
                crs=src.crs,
                attribution=f"© OpenStreetMap contributors, CRS: {src.crs.to_string()}"
            )
            show(
                relativeDeviation,
                transform=src.transform,
                ax=ax4,
                cmap=cmocean.cm.balance_r,
                vmin=-1,
                vmax=1
            )
            cbar4 = plt.colorbar(
                plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-1, vmax=1), cmap=cmocean.cm.balance_r),
                ax=ax4,
                shrink=0.8
            )
            cbar4.set_label("relative deviation")

            ax1.annotate('a)', xy=(0.03, 0.1), xycoords='axes fraction')
            ax2.annotate('b)', xy=(0.03, 0.1), xycoords='axes fraction')
            ax3.annotate('c)', xy=(0.03, 0.1), xycoords='axes fraction')
            ax4.annotate('d)', xy=(0.03, 0.1), xycoords='axes fraction')
            # Add datetime as title
            fig.suptitle(imfiles["date"][i].strftime("%Y-%m-%d"))
            fig.tight_layout()
            # Save the figure
            fig.savefig(
                os.path.join(imoutfolder, imfiles["imname"].values[i].replace(".tif", ".png")),
                dpi=300,
                bbox_inches="tight"
            )
            plt.close(fig)
            print(f"Processed file {imfiles['impath'].values[i]}")

    except Exception as e:
        print(f"Error processing file {imfiles['impath'].values[i]}: {e}")




# # Filter for June, July, and August only
# imfiles["year"] = imfiles["date"].dt.year
# imfiles["month"] = imfiles["date"].dt.month
# imfiles = imfiles[imfiles["month"].isin([6, 7, 8])]

# # %%
# # Select files for the year of interest
# year_of_interest = 2023
# imfiles_selected = imfiles[imfiles["year"] == year_of_interest].sort_values("date")

# # Create time variable for coordinates
# time_coords = imfiles_selected["date"].dt.strftime("%Y-%m-%d").values

# # Open and process raster files
# ds_list = []
# for file_path, date in zip(imfiles_selected["impath"].values, time_coords):
#     try:
#         # Open the raster file using xarray with the rasterio engine
#         single_ds = xr.open_dataset(file_path, engine="rasterio")
#         # Ensure band 5 exists and select it
#         if single_ds.rio.count >= 5:
#             band5 = single_ds.isel(band=4)  # Band 5 (zero-based index)
#             # Assign time coordinate
#             band5 = band5.assign_coords(time=date).expand_dims("time")
#             ds_list.append(band5)
#     except Exception as e:
#         print(f"Error processing file {file_path}: {e}")

# # Combine all datasets along the time dimension
# if ds_list:
#     ds = xr.concat(ds_list, dim="time")
#     print("Dataset successfully created.")
# else:
#     print("No valid files with band 5 found.")

# %%
