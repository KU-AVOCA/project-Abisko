"""
Analyzes and visualizes defoliation patterns using remote sensing data.
This script performs the following main operations:
1.  Loads annual remote sensing data (presumably a vegetation index or similar)
    and calculates its standard deviation over time.
2.  For selected years (2013, 2014, 2021-2024), it processes the data to
    highlight significant negative deviations (potential defoliation). This involves:
    - Filtering out values with absolute deviations less than the overall standard deviation.
    - Retaining only negative deviations.
    - Applying a water mask to exclude water bodies.
3.  Visualizes the processed defoliation data for these selected years on a 3x2
    grid of maps. Each map includes:
    - The defoliation intensity raster.
    - A basemap for geographical context.
    - A Point of Interest (POI) marker.
    - A common colorbar indicating defoliation intensity.
4.  Calculates annual defoliation statistics for the period 2013-2024:
    - Mean defoliation intensity.
    - Standard deviation of defoliation intensity.
    - Defoliation extent (ratio of defoliated pixels to total non-water pixels).
5.  Generates two summary plots:
    - A boxplot showing the distribution of defoliation intensity for each year.
    - A barplot showing the annual defoliation extent, with bars colored by
      the mean defoliation intensity for that year.

Shunan Feng (shf@ign.ku.dk)
"""
#%%
import pandas as pd
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

# %% visualize the rd_annual for years 2013, 2014, 2021-2024
years = np.arange(2013, 2025)

rd_2013 = rd_annual[np.where(years == 2013)[0][0], :, :]
rd_2013 = np.where(np.abs(rd_2013) < rd_std, np.nan, rd_2013)
rd_2013[rd_2013 > 0] = np.nan

rd_2014 = rd_annual[np.where(years == 2014)[0][0], :, :]
rd_2014 = np.where(np.abs(rd_2014) < rd_std, np.nan, rd_2014)
rd_2014[rd_2014 > 0] = np.nan

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
    # Apply watermask to all years
    rd_2013[watermask == 0] = np.nan
    rd_2014[watermask == 0] = np.nan
    rd_2021[watermask == 0] = np.nan
    rd_2022[watermask == 0] = np.nan
    rd_2023[watermask == 0] = np.nan
    rd_2024[watermask == 0] = np.nan
poi = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=[19.0524377], y=[68.34836531]), crs="EPSG:4326").to_crs(src.crs)

# Create a figure with 6 subplots in a 3x2 grid
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 2)

# 2013
ax1 = fig.add_subplot(gs[0, 0])
show(
    rd_2013,
    transform=src.transform,
    ax=ax1,
    cmap=cmcrameri.cm.lajolla,
    vmin=-1,
    vmax=0,
)
ctx.add_basemap(
    ax1,
    attribution=f"© OpenStreetMap contributors, CRS: {src.crs}",
    crs=src.crs,
    attribution_size=15
)
show(
    rd_2013,
    transform=src.transform,
    ax=ax1,
    cmap=cmcrameri.cm.lajolla,
    vmin=-1,
    vmax=0,
)
# overlay poi
poi.plot(ax=ax1, color="k", markersize=30)
ax1.annotate("E1", xy=(poi.geometry.x[0], poi.geometry.y[0]), 
             xytext=(10, 10), textcoords="offset points",
             color="black")

# 2014
ax2 = fig.add_subplot(gs[0, 1])
show(
    rd_2014,
    transform=src.transform,
    ax=ax2,
    cmap=cmcrameri.cm.lajolla,
    vmin=-1,
    vmax=0,
)
ctx.add_basemap(
    ax2,
    attribution=f"© OpenStreetMap contributors, CRS: {src.crs}",
    crs=src.crs,
    attribution_size=15
)
show(
    rd_2014,
    transform=src.transform,
    ax=ax2,
    cmap=cmcrameri.cm.lajolla,
    vmin=-1,
    vmax=0,
)
poi.plot(ax=ax2, color="k", markersize=30)
ax2.annotate("E1", xy=(poi.geometry.x[0], poi.geometry.y[0]),
             xytext=(10, 10), textcoords="offset points",
             color="black")

# 2021
ax3 = fig.add_subplot(gs[1, 0])
show(
    rd_2021,
    transform=src.transform,
    ax=ax3,
    cmap=cmcrameri.cm.lajolla,
    vmin=-1,
    vmax=0,
)
ctx.add_basemap(
    ax3,
    attribution=f"© OpenStreetMap contributors, CRS: {src.crs}",
    crs=src.crs,
    attribution_size=15
)
show(
    rd_2021,
    transform=src.transform,
    ax=ax3,
    cmap=cmcrameri.cm.lajolla,
    vmin=-1,
    vmax=0,
)
poi.plot(ax=ax3, color="k", markersize=30)
ax3.annotate("E1", xy=(poi.geometry.x[0], poi.geometry.y[0]),
             xytext=(10, 10), textcoords="offset points",
             color="black")

# 2022
ax4 = fig.add_subplot(gs[1, 1])
show(
    rd_2022,
    transform=src.transform,
    ax=ax4,
    cmap=cmcrameri.cm.lajolla,
    vmin=-1,
    vmax=0,
)
ctx.add_basemap(
    ax4,
    attribution=f"© OpenStreetMap contributors, CRS: {src.crs}",
    crs=src.crs,
    attribution_size=15
)
show(
    rd_2022,
    transform=src.transform,
    ax=ax4,
    cmap=cmcrameri.cm.lajolla,
    vmin=-1,
    vmax=0,
)
poi.plot(ax=ax4, color="k", markersize=30)
ax4.annotate("E1", xy=(poi.geometry.x[0], poi.geometry.y[0]),
             xytext=(10, 10), textcoords="offset points",
             color="black")

# 2023
ax5 = fig.add_subplot(gs[2, 0])
show(
    rd_2023,
    transform=src.transform,
    ax=ax5,
    cmap=cmcrameri.cm.lajolla,
    vmin=-1,
    vmax=0,
)
ctx.add_basemap(
    ax5,
    attribution=f"© OpenStreetMap contributors, CRS: {src.crs}",
    crs=src.crs,
    attribution_size=15
)
show(
    rd_2023,
    transform=src.transform,
    ax=ax5,
    cmap=cmcrameri.cm.lajolla,
    vmin=-1,
    vmax=0,
)
poi.plot(ax=ax5, color="k", markersize=30)
ax5.annotate("E1", xy=(poi.geometry.x[0], poi.geometry.y[0]),
             xytext=(10, 10), textcoords="offset points",
             color="black")

# 2024
ax6 = fig.add_subplot(gs[2, 1])
show(
    rd_2024,
    transform=src.transform,
    ax=ax6,
    cmap=cmcrameri.cm.lajolla,
    vmin=-1,
    vmax=0,
)
ctx.add_basemap(
    ax6,
    attribution=f"© OpenStreetMap contributors, CRS: {src.crs}",
    crs=src.crs,
    attribution_size=15
)
show(
    rd_2024,
    transform=src.transform,
    ax=ax6,
    cmap=cmcrameri.cm.lajolla,
    vmin=-1,
    vmax=0,
)
poi.plot(ax=ax6, color="k", markersize=30)
ax6.annotate("E1", xy=(poi.geometry.x[0], poi.geometry.y[0]),
             xytext=(10, 10), textcoords="offset points",
             color="black")

# Add colorbar for all subplots at the bottom
cbar_ax = fig.add_axes([0.15, -0.01, 0.7, 0.02])  # [left, bottom, width, height]
norm = plt.Normalize(vmin=-1, vmax=0)
cmap = cmcrameri.cm.lajolla
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label(r'Defoliation Intensity (Relative Deviation < -1$\sigma$)')

# Scientific notation for all axes
ax1.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
ax2.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
ax3.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
ax4.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
ax5.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
ax6.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))

# Add titles to the subplots
ax1.set_title('a) 2013')
ax2.set_title('b) 2014')
ax3.set_title('c) 2021')
ax4.set_title('d) 2022')
ax5.set_title('e) 2023')
ax6.set_title('f) 2024')

plt.tight_layout()
fig.savefig("../print/relative_deviation_2013-2014-2021-2024.png", dpi=300, bbox_inches="tight")
fig.savefig("../print/relative_deviation_2013-2014-2021-2024.pdf", dpi=300, bbox_inches="tight")

# %% annual defoliation extent ratio, mean, and std
df_defoliation = pd.DataFrame()
years = np.arange(2013, 2025)
rd_boxplot_data = []

for i in range(len(years)):
    rd = rd_annual[i, :, :]
    rd = np.where(np.abs(rd) < rd_std, np.nan, rd)
    rd[rd >= 0] = np.nan
    rd[watermask == 0] = np.nan
    df_defoliation.loc[i, "year"] = int(years[i])
    df_defoliation.loc[i, "mean"] = np.nanmean(rd)
    df_defoliation.loc[i, "std"] = np.nanstd(rd)
    df_defoliation.loc[i, "extent"] = np.sum(~np.isnan(rd)) / np.size(rd) if np.size(rd) > 0 else 0
    
    # Collect data for boxplot
    rd_boxplot_data.append(rd[~np.isnan(rd)].flatten())

# Plot boxplot for each year
fig, ax = plt.subplots(figsize=(12, 6))
ax.boxplot(rd_boxplot_data, labels=years, showfliers=False)
ax.set_title("Defoliation Intensity Distribution by Year")
ax.set_xlabel("Year")
ax.set_ylabel("Defoliation Intensity (Relative Deviation)")


fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    data=df_defoliation,
    x="year",
    y="extent",
    hue="mean",
    palette="crest",
    ax=ax,
)
# %%
