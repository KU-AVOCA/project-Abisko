#%%
import matplotlib.pyplot as plt
import rasterio
import numpy as np
import cartopy.crs as ccrs
from rasterio.plot import show
import seaborn as sns
import contextily as ctx
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib import colormaps
sns.set_theme(style="darkgrid", font_scale=1.5)
#%%
def create_combined_figure():
    # Create a figure with custom layout
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, height_ratios=[2, 1])
    
    # PART 1: Map plot (first row, spans both columns)
    ax_map = fig.add_subplot(gs[0, :])
    
    tif_path = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/CCDC/data/numBreaks.tif"
    with rasterio.open(tif_path) as src:
        # Read the raster data
        raster = src.read(1)
        raster[raster == 0] = np.nan
        
        # Handle nodata values
        if src.nodata is not None:
            raster = np.ma.masked_equal(raster, src.nodata)
        
        # Define the number of breaks
        num_breaks = int(np.nanmax(raster))
        
        # Create a colormap with discrete colors
        cmap = colormaps.get_cmap('viridis')
        
        # Plot raster with the discrete colormap
        show(raster, transform=src.transform, ax=ax_map, cmap=cmap, vmin=1, vmax=num_breaks)
        
        # Add XYZ tile basemap
        ctx.add_basemap(
            ax_map, 
            # attribution=f"© OpenStreetMap contributors, CRS: {src.crs}",
            crs=ccrs.UTM(34)
        )
        show(raster, transform=src.transform, ax=ax_map, cmap=cmap, vmin=1, vmax=num_breaks)
        
        # Add colorbar with discrete intervals
        ticks = np.arange(1, num_breaks + 1)
        norm = mcolors.BoundaryNorm(ticks - 0.5, cmap.N)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=ax_map,
                        ticks=ticks,
                        shrink=0.5)
        cbar.set_label('Number of Breaks')
        
        # Format x-axis tick labels using scientific notation
        ax_map.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
        ax_map.text(0.7, 0.1, f"a) CRS: {src.crs}", transform=ax_map.transAxes)
        # ax_map.set_title('CCDC Break Detection Map', fontsize=14)
    
    # PART 2: Number of breaks histogram (bottom left)
    ax_hist1 = fig.add_subplot(gs[1, 0])
    
    with rasterio.open(tif_path) as src:
        raster = src.read(1)
        raster[raster == 0] = np.nan
        breaks = raster[~np.isnan(raster)].flatten()
        
        # Create a DataFrame for the breaks
        breaks_df = pd.DataFrame({'breaks': breaks})
        
        # Plot histogram
        sns.histplot(
            data=breaks_df,
            x='breaks',
            binwidth=1,
            binrange=(breaks_df['breaks'].min() - 0.5, breaks_df['breaks'].max() + 0.5),
            ax=ax_hist1
        )
        ax_hist1.set_xlabel('Number of Breaks')
        ax_hist1.set_ylabel('Frequency')
        ax_hist1.text(0.9, 0.1, 'b)', transform=ax_hist1.transAxes)
        # ax_hist1.set_title('Histogram of Number of Breaks')
    
    # PART 3: Change time histogram (bottom right)
    ax_hist2 = fig.add_subplot(gs[1, 1])
    
    # Load and process the change time data
    first_change_time_path = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/CCDC/data/firstChangeTime.tif"
    last_change_time_path = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/CCDC/data/lastChangeTime.tif"
    
    # Read and process the firstChangeTime raster
    with rasterio.open(first_change_time_path) as src:
        first_change_time = src.read(1)
        first_change_time[first_change_time == 0] = np.nan
    
    # Read and process the lastChangeTime raster
    with rasterio.open(last_change_time_path) as src:
        last_change_time = src.read(1)
        last_change_time[last_change_time == 0] = np.nan
    
    # Flatten the arrays and remove NaN values
    first_change_years = first_change_time[~np.isnan(first_change_time)].flatten()
    last_change_years = last_change_time[~np.isnan(last_change_time)].flatten()
    
    # Create DataFrames
    first_change_df = pd.DataFrame({
        'time': first_change_years,
        'legend': 'first_change_years'
    })
    
    last_change_df = pd.DataFrame({
        'time': last_change_years,
        'legend': 'last_change_years'
    })
    
    # Combine both DataFrames
    change_years_df = pd.concat([first_change_df, last_change_df], ignore_index=True)
    
    # Plot histogram
    sns.histplot(
        data=change_years_df,
        x='time',
        hue='legend',
        binwidth=1,
        ax=ax_hist2
    )
    ax_hist2.set_xlabel('Year')
    ax_hist2.set_ylabel('Frequency')
    ax_hist2.text(0.95, 0.1, 'c)', transform=ax_hist2.transAxes)
    # ax_hist2.set_title('First and Last Change Time Comparison')
    # Adjust layout and show figure
    plt.tight_layout()
    return fig

# Display the combined figure
combined_fig = create_combined_figure()
combined_fig.savefig('../../print/ccdc_breaks.png', dpi=300, bbox_inches='tight')
combined_fig.savefig('../../print/ccdc_breaks.pdf', dpi=300, bbox_inches='tight')
#%%
