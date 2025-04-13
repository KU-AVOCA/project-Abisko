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
import geopandas as gpd
from matplotlib.gridspec import GridSpec
import seaborn as sns
sns.set_theme(style="darkgrid", font_scale=1.5)

# %%
imfolder = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/CCDC/data/"
rd_annual = np.load(os.path.join(imfolder, "rd_annual.npy"))
rd_std = np.nanstd(rd_annual, axis=0)

# %% visualize the rd_annual for years 2021-2024
years = np.arange(2014, 2025)
rd_2021 = rd_annual[np.where(years == 2021)[0][0], :, :]
rd_2021 = np.where(np.abs(rd_2021) < rd_std, np.nan, rd_2021)
rd_2021[rd_2021 > 0] = np.nan

rd_2022 = rd_annual[np.where(years == 2022)[0][0], :, :]
rd_2022 = np.where(np.abs(rd_2022) < rd_std, np.nan, rd_2022)
rd_2022[rd_2022 > 0] = np.nan

rd_2023 = rd_annual[np.where(years == 2023)[0][0], :, :]
rd_2023 = np.where(np.abs(rd_2023) < rd_std, np.nan, rd_2023)
rd_2023[rd_2023 > 0] = np.nan

rd_2024 = rd_annual[np.where(years == 2024)[0][0], :, :]
rd_2024 = np.where(np.abs(rd_2024) < rd_std, np.nan, rd_2024)
rd_2024[rd_2024 > 0] = np.nan
# %%
with rio.open(os.path.join(imfolder, "waterMask.tif")) as src:
    watermask = src.read(1)
    rd_2021[watermask == 0] = np.nan
    rd_2022[watermask == 0] = np.nan
    rd_2023[watermask == 0] = np.nan
    rd_2024[watermask == 0] = np.nan
poi = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=[19.0524377], y=[68.34836531]), crs="EPSG:4326").to_crs(src.crs)

fig = plt.figure(figsize=(16, 8))
gs = GridSpec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
show(
    rd_2021,
    transform=src.transform,
    ax=ax1,
    cmap=cmcrameri.cm.lajolla, #cmocean.cm.balance_r,
    vmin=-1,
    vmax=0,
)
ctx.add_basemap(
    ax1,
    attribution=f"© OpenStreetMap contributors, CRS: {src.crs}",
    crs=src.crs,
    attribution_size=15  # Adjust the size as needed
)
show(
    rd_2021,
    transform=src.transform,
    ax=ax1,
    cmap=cmcrameri.cm.lajolla, #cmocean.cm.balance_r,
    vmin=-1,
    vmax=0,
)
# overlay poi
poi.plot(ax=ax1, color="k", markersize=30)
ax1.annotate("E1", xy=(poi.geometry.x[0], poi.geometry.y[0]), 
             xytext=(10, 10), textcoords="offset points",
             color="black")

ax2 = fig.add_subplot(gs[0, 1])
show(
    rd_2022,
    transform=src.transform,
    ax=ax2,
    cmap=cmcrameri.cm.lajolla, #cmocean.cm.balance_r,
    vmin=-1,
    vmax=0,
)
ctx.add_basemap(
    ax2,
    attribution=f"© OpenStreetMap contributors, CRS: {src.crs}",
    crs=src.crs,
    attribution_size=15  # Adjust the size as needed
)
show(
    rd_2022,
    transform=src.transform,
    ax=ax2,
    cmap=cmcrameri.cm.lajolla, #cmocean.cm.balance_r,
    vmin=-1,
    vmax=0,
)
#
poi.plot(ax=ax2, color="k", markersize=30)
ax2.annotate("E1", xy=(poi.geometry.x[0], poi.geometry.y[0]),
             xytext=(10, 10), textcoords="offset points",
             color="black")
ax3 = fig.add_subplot(gs[1, 0])
show(
    rd_2023,
    transform=src.transform,
    ax=ax3,
    cmap=cmcrameri.cm.lajolla, #cmocean.cm.balance_r,
    vmin=-1,
    vmax=0,
)
ctx.add_basemap(
    ax3,
    attribution=f"© OpenStreetMap contributors, CRS: {src.crs}",
    crs=src.crs,
    attribution_size=15  # Adjust the size as needed
)
show(
    rd_2023,
    transform=src.transform,
    ax=ax3,
    cmap=cmcrameri.cm.lajolla, #cmocean.cm.balance_r,
    vmin=-1,
    vmax=0,
)
# overlay poi
poi.plot(ax=ax3, color="k", markersize=30)
ax3.annotate("E1", xy=(poi.geometry.x[0], poi.geometry.y[0]),
             xytext=(10, 10), textcoords="offset points",
             color="black")
ax4 = fig.add_subplot(gs[1, 1])
show(
    rd_2024,
    transform=src.transform,
    ax=ax4,
    cmap=cmcrameri.cm.lajolla, #cmocean.cm.balance_r,
    vmin=-1,
    vmax=0,
)
ctx.add_basemap(
    ax4,
    attribution=f"© OpenStreetMap contributors, CRS: {src.crs}",
    crs=src.crs,
    attribution_size=15  # Adjust the size as needed
)
show(
    rd_2024,
    transform=src.transform,
    ax=ax4,
    cmap=cmcrameri.cm.lajolla, #cmocean.cm.balance_r,
    vmin=-1,
    vmax=0,
)
# overlay poi
poi.plot(ax=ax4, color="k", markersize=30)
ax4.annotate("E1", xy=(poi.geometry.x[0], poi.geometry.y[0]),
             xytext=(10, 10), textcoords="offset points",
             color="black")

# Add colorbar for all subplots at the bottom
cbar_ax = fig.add_axes([0.15, 0.03, 0.7, 0.02])  # [left, bottom, width, height]
norm = plt.Normalize(vmin=-1, vmax=0)
cmap = cmcrameri.cm.lajolla #cmocean.cm.balance_r
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label(r'Defoliation Intensity (Relative Deviation < -1$\sigma$)')

ax1.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
ax2.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
ax3.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
ax4.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))

# # Add titles to the subplots
ax1.set_title('a) 2021')
ax2.set_title('b) 2022')
ax3.set_title('c) 2023')
ax4.set_title('d) 2024')

fig.savefig("../print/relative_deviation_2021-2024.png", dpi=300, bbox_inches="tight")
fig.savefig("../print/relative_deviation_2021-2024.pdf", dpi=300, bbox_inches="tight")

# %%
