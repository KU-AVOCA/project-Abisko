'''
Test script to visualize the drone images and calculate the statistics
for the relative difference.

Shunan Feng (shf@ign.ku.dk)
'''
#%%
# import pandas as pd
import numpy as np
# import glob
import os
import matplotlib.pyplot as plt
import contextily as ctx
import rasterio as rio
from rasterio.plot import show
# import cmocean
import cmcrameri
# import geopandas as gpd
from matplotlib.gridspec import GridSpec
import seaborn as sns
sns.set_theme(style="darkgrid", font_scale=1.5)

# %%
imfolder = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/drone/indices/"
imgcc = os.path.join(imfolder, "21_07_01_orthomosaic_georef_processed_GCC.tif")
img2022 = os.path.join(imfolder, "22_07_04_relative_difference.tif")
img2023 = os.path.join(imfolder, "23_07_04_relative_difference.tif")

#%% statistics
with rio.open(img2022) as src:
    img = src.read(1)
    img[img<-1] = np.nan
    img[img>=0] = np.nan
    # img = np.where(np.logical_or(img < -1, img >=0), np.nan, img)
    crs = src.crs
    transform = src.transform
print(f"2022: {np.nanmean(img):.4f} +/- {np.nanstd(img):.4f}, extent: {~np.isnan(img).sum()}")
with rio.open(img2023) as src:
    img = src.read(1)
    img = np.where(np.logical_or(img < -1, img >=0), np.nan, img)
    crs = src.crs
    transform = src.transform
print(f"2023: {np.nanmean(img):.4f} +/- {np.nanstd(img):.4f}, extent: {~np.isnan(img).sum()}")

#%%
fig = plt.figure(figsize=(8, 16))
gs = GridSpec(2, 1)
ax1 = fig.add_subplot(gs[0, 0])

with rio.open(img2022) as src:
    img = src.read(1)
    img[img<-1] = np.nan
    img[img>=0] = np.nan
    # img = np.where(np.logical_or(img < -1, img >=0), np.nan, img)
    crs = src.crs
    transform = src.transform
show(
    img,
    transform=transform,
    ax=ax1,
    cmap=cmcrameri.cm.lajolla, #cmocean.cm.balance_r,
    vmin=-1,
    vmax=0,
)
ctx.add_basemap(
    ax1,
    attribution=f"© OpenStreetMap contributors, CRS: {crs}",
    crs=crs,
    attribution_size=15  # Adjust the size as needed
)
show(
    img,
    transform=transform,
    ax=ax1,
    cmap=cmcrameri.cm.lajolla, #cmocean.cm.balance_r,   
    vmin=-1,
    vmax=0,
)
ax2 = fig.add_subplot(gs[1, 0])
with rio.open(img2023) as src:
    img = src.read(1)
    img = np.where(np.logical_or(img < -1, img >=0), np.nan, img)
    crs = src.crs
    transform = src.transform
show(
    img,
    transform=transform,
    ax=ax2,
    cmap=cmcrameri.cm.lajolla, #cmocean.cm.balance_r,
    vmin=-1,
    vmax=0,
)
ctx.add_basemap(
    ax2,
    attribution=f"© OpenStreetMap contributors, CRS: {crs}",
    crs=crs,
    attribution_size=15  # Adjust the size as needed
)
show(
    img,
    transform=transform,
    ax=ax2,
    cmap=cmcrameri.cm.lajolla, #cmocean.cm.balance_r,
    vmin=-1,
    vmax=0,
)
cbar_ax = fig.add_axes([0.15, 0.03, 0.7, 0.02])  # [left, bottom, width, height]
norm = plt.Normalize(vmin=-1, vmax=0)
cmap = cmcrameri.cm.lajolla #cmocean.cm.balance_r
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Relative Deviation')
ax1.annotate("a)", xy=(0.5, 0.95), xycoords="axes fraction", ha="center")
ax2.annotate("b)", xy=(0.5, 0.95), xycoords="axes fraction", ha="center")
