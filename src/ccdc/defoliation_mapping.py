#%%
# import pandas as pd
import numpy as np
# import glob
import os
import matplotlib.pyplot as plt
import contextily as ctx
import rasterio as rio
from rasterio.plot import show
import cmocean
# import cmcrameri
from matplotlib.gridspec import GridSpec
import seaborn as sns
sns.set_theme(style="darkgrid", font_scale=1.5)

# %%
imfolder = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/CCDC/data/"
cs_annual = np.load(os.path.join(imfolder, "cs_annual.npy"))
rd_annual = np.load(os.path.join(imfolder, "rd_annual.npy"))
rd_std = np.nanstd(rd_annual, axis=0)

#%%
years = np.arange(2014, 2025)
for i in range(len(years)):
    year = years[i]
    print(f"Processing year {year}...")
    cs_image = cs_annual[i]
    rd_image = rd_annual[i]
    rd_image = np.where(np.abs(rd_image) < rd_std, np.nan, rd_image)
    with rio.open(os.path.join(imfolder, "waterMask.tif")) as src:
        watermask = src.read(1)
        rd_image[watermask == 0] = np.nan
        cs_image[watermask == 0] = np.nan
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    show(
        cs_image,
        transform=src.transform,
        ax=ax1,
        cmap=cmocean.cm.balance_r,
        vmin=-4,
        vmax=4,
    )
    ctx.add_basemap(
        ax1,
        # attribution=f"© OpenStreetMap contributors, CRS: {src.crs}",
        crs=src.crs
    )
    show(
        cs_image,
        transform=src.transform,
        ax=ax1,
        cmap=cmocean.cm.balance_r,
        vmin=-4,
        vmax=4,
    )
    cbar1 = plt.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-4, vmax=4), cmap=cmocean.cm.balance_r),
        ax=ax1,
        shrink=0.5,
        extend="both"
    )
    cbar1.set_label("Condition Score")
    ax2 = fig.add_subplot(gs[0, 1])
    show(
        rd_image,
        transform=src.transform,
        ax=ax2,
        cmap=cmocean.cm.balance_r,
        vmin=-1,
        vmax=1
    )
    ctx.add_basemap(
        ax2,
        # attribution=f"© OpenStreetMap contributors, CRS: {src.crs}",
        crs=src.crs
    )
    show(
        rd_image,
        transform=src.transform,
        ax=ax2,
        cmap=cmocean.cm.balance_r,
        vmin=-1,
        vmax=1
    )
    cbar2 = plt.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-1, vmax=1), cmap=cmocean.cm.balance_r),
        ax=ax2,
        shrink=0.5,
        # extend="both"
    )
    cbar2.set_label("Relative Deviation")
    # Save the figure
    fig.savefig(os.path.join(imfolder, f"cs_rd_{year}.png"), dpi=300, bbox_inches="tight")
    # fig.savefig(os.path.join(imfolder, f"cs_rd_{year}.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    # Calculate the mean for each year