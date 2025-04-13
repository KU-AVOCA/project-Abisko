'''
Insect Analysis Script for Abisko Project
This script analyzes data from multiple sources to examine relationships between
vegetation greenness, insect abundance, and BVOC (Biogenic Volatile Organic Compounds)
emissions in Abisko, Sweden. The script performs the following functions:

1. Loads and processes CCDC (Continuous Change Detection and Classification) data for GCC metrics
2. Processes insect abundance survey data from 2022-2023
3. Determines daylight hours based on solar elevation angles using pvlib
4. Processes tower RGB camera images, filtering out night/low-light conditions
5. Analyzes BVOC emission data with focus on Empetrum species
6. Creates comprehensive time series visualizations showing:
    - CCDC-derived GCC data (fitted, predicted, and observed) with insect abundance
    - Tower camera greenness ratio with insect abundance
    - BVOC emissions with insect abundance
7. Exports high-quality figures in PNG and PDF formats

The script integrates multiple data streams to investigate relationships between 
vegetation phenology, insect populations, and plant volatile emissions across
multiple years (2021-2023) in the Abisko region.

Author: Shunan Feng (shf@ign.ku.dk)
'''
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pvlib

from scipy import stats
sns.set_theme(style="darkgrid", font_scale=1.5)

#%% CCDC Data
df_origional = pd.read_csv("/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/CCDC/data/E1_GCC_origional.csv")
df_predicted = pd.read_csv("/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/CCDC/data/E1_GCC_predicted.csv")
df_origional[df_origional.GCC >1] = np.nan
df_origional[df_origional.GCC <0] = np.nan

#data preprocessing 
df_origional = df_origional.rename(columns={"system:time_start": "date"})
df_predicted = df_predicted.rename(columns={"system:time_start": "date"})
df_origional["date"] = pd.to_datetime(df_origional["date"], format="%d-%b-%y")
df_predicted["date"] = pd.to_datetime(df_predicted["date"], format="%d-%b-%y")
df_origional = df_origional.groupby(pd.Grouper(key="date", freq="d"))["GCC"].mean().reset_index()
df_ccdc = pd.merge(
    df_origional,
    df_predicted,
    on=["date"]
    # suffixes=("_origional", "_predicted"),
)
df_ccdc["month"] = df_ccdc["date"].dt.month
df_ccdc["year"] = df_ccdc["date"].dt.year
df_ccdc["day"] = df_ccdc["date"].dt.day
df_ccdc = df_ccdc[df_ccdc['year'].isin([2021, 2022, 2023])]

#%% read insect abundance data
df_insect = pd.read_excel('/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/Insect_Survey/Insect_Surveys_2022-2023_Abisko.xlsx', sheet_name='all')
df_insect['date'] = pd.to_datetime(df_insect['date'], errors='coerce')
df_insect['year'] = df_insect['date'].dt.year
df_insect['month'] = df_insect['date'].dt.month
df_insect['day'] = df_insect['date'].dt.day
df_insect['doys'] = df_insect['date'].dt.dayofyear
# Group insect abundance data by date and calculate mean and standard deviation
df_insect_grouped = df_insect.groupby('date').agg({
    'total': ['mean', 'std']
}).reset_index()

# Flatten column names
df_insect_grouped.columns = ['date', 'total_mean', 'total_std']
# sns.barplot(data=df_insect, x='doys', y='total', hue='year', errorbar='sd')

#%% Define Abisko coordinates
SITE_LATITUDE = 68.34808742 # in decimal degrees 
SITE_LONGITUDE = 19.05077561 # in decimal degrees
SITE_ELEVATION = 400  # meters above sea level (approximate)

#%% Function to determine daylight hours
def is_daytime(row, min_elevation=5.0):
    """
    Determine if a timestamp is during daylight hours based on solar elevation.
    
    Args:
        row: DataFrame row containing 'datetime'
        min_elevation: Minimum solar elevation angle (in degrees) to be considered daytime
                      (5 degrees excludes dawn/dusk periods)
    
    Returns:
        bool: True if the timestamp is during daylight hours
    """
    try:
        # Get datetime from row
        timestamp = row['datetime']
        if pd.isna(timestamp):
            return False
        
        # Calculate solar position
        solpos = pvlib.solarposition.get_solarposition(
            timestamp, 
            SITE_LATITUDE, 
            SITE_LONGITUDE,
            altitude=SITE_ELEVATION
        )
        
        # Get solar elevation angle
        elevation = solpos['elevation'].iloc[0]
        
        # Check if it's daytime (sun is above the minimum elevation)
        return elevation > min_elevation
        
    except Exception as e:
        print(f"Error calculating solar position: {e}")
        return False

#%% Tower Time Lapse Image Data
csvfile = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/Tower RGB images/Data_greenessByShunan_kmeans_mean/results/green_ratio_kmeans.csv'
df_tower = pd.read_csv(csvfile)

# Rename columns for clarity
df_tower.rename(
    columns={
        'class1_ratio': 'understory_ratio',
        'class1_mean': 'understory_mean',
        'class1_std': 'understory_std',
        'class1_norm': 'understory_norm',
        'class2_ratio': 'birch_ratio',
        'class2_mean': 'birch_mean',
        'class2_std': 'birch_std',
        'class2_norm': 'birch_norm'
    },
    inplace=True
)

# Convert datetime and extract components
df_tower['datetime'] = pd.to_datetime(df_tower['datetime'], errors='coerce')
df_tower['year'] = df_tower['datetime'].dt.year
df_tower['month'] = df_tower['datetime'].dt.month
df_tower['doys'] = df_tower['datetime'].dt.dayofyear
df_tower['hour'] = df_tower['datetime'].dt.hour
df_tower['minute'] = df_tower['datetime'].dt.minute
df_tower['imgroup'] = df_tower['filename'].str.split('/').str[-3]

# Filter out night/dusk/dawn images
print("Total images before filtering:", len(df_tower))
df_tower['is_daytime'] = df_tower.apply(is_daytime, axis=1)
daytime_df_tower = df_tower[df_tower['is_daytime']]
# remove images taken after 2023-08-17 in west-facing camera due to overexposure
daytime_df_tower = daytime_df_tower[~((daytime_df_tower['datetime'] > pd.to_datetime("2023-08-17")) & (daytime_df_tower['imgroup'].str.contains('West')))]
print("Daytime images:", len(daytime_df_tower))
print(f"Removed {len(df_tower) - len(daytime_df_tower)} images taken during night or low-light conditions")
# only keep west-facing images 
west_df = daytime_df_tower[daytime_df_tower['imgroup'].str.contains('West')]
west_df['date'] = west_df['datetime'].dt.date

#%% BVOC Data
df_bvoc = pd.read_csv('/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/BVOC/Data_Master_Abisko_2022-2023_V4.csv',
                      delimiter=';')
df_bvoc['date'] = pd.to_datetime(df_bvoc['Date'], errors='coerce')
df_bvoc['year'] = df_bvoc['date'].dt.year
df_bvoc['month'] = df_bvoc['date'].dt.month
df_bvoc['day'] = df_bvoc['date'].dt.day
df_bvoc['doys'] = df_bvoc['date'].dt.dayofyear

df_bvoc_emp = df_bvoc[df_bvoc['Species'] == 'Emp.']
# Group BVOC data by date to get daily means and standard deviations
df_bvoc_emp_daily = df_bvoc_emp.groupby(['date']).agg({
    'Total_BVOC': ['mean', 'std', 'count']
}).reset_index()

# Flatten the multi-level columns
df_bvoc_emp_daily.columns = ['date', 'Total_BVOC_mean', 'Total_BVOC_std', 'Total_BVOC_count']

# # Replace NaN values in std column with 0 (for dates with only one measurement)
# df_bvoc_daily['Total_BVOC_std'] = df_bvoc_daily['Total_BVOC_std'].fillna(0)

# # Extract year, month, day and day of year for plotting
# df_bvoc_daily['year'] = df_bvoc_daily['date'].dt.year
# df_bvoc_daily['month'] = df_bvoc_daily['date'].dt.month
# df_bvoc_daily['day'] = df_bvoc_daily['date'].dt.day
# df_bvoc_daily['doys'] = df_bvoc_daily['date'].dt.dayofyear

# # Create separate dataframes for each species if needed
# df_bvoc_emp_daily = df_bvoc_daily[df_bvoc_daily['Species'] == 'Emp.']
# df_bvoc_vac_daily = df_bvoc_daily[df_bvoc_daily['Species'] == 'Vac.']
# sns.lineplot(data=df_bvoc, x='doys', y='Total_BVOC', hue='year', errorbar='sd', style='Species')

#%% compare insect abundance data with CCDC data and tower camera greenness ratio
fig, (ax1, ax3, ax5) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# First subplot: CCDC data with insect abundance
sns.lineplot(data=df_ccdc[df_ccdc.year<2022], x='date', y='GCC_predicted', ax=ax1, label='GCC fitted', color=sns.palettes.color_palette()[0])
sns.lineplot(data=df_ccdc[df_ccdc.year>=2022], x='date', y='GCC_predicted', ax=ax1, label='GCC predicted', dashes=(2, 2), color=sns.palettes.color_palette()[1])
sns.scatterplot(data=df_ccdc, x='date', y='GCC', ax=ax1, label='GCC observed', color=sns.palettes.color_palette()[2])
ax1.set(
    ylim=(0.25, 0.5), 
    xlim=(pd.to_datetime("2021-01-01"), pd.to_datetime("2023-12-31")),
    ylabel='Green Chromatic Coordinate\n(GCC)',)
ax2 = ax1.twinx()
ax2.grid(False)
# Plot insect abundance data with error bars on first subplot
ax2.errorbar(
    x=df_insect_grouped['date'],
    y=df_insect_grouped['total_mean'],
    yerr=df_insect_grouped['total_std'],
    color="#734118",
    marker='o',
    linestyle='',
    label='Insect Abundance (mean ± sd)',
)
ax2.set(
    ylabel='Insect Abundance',
    ylim=(-5, 170),
)

# Second subplot: Tower greenness ratio with insect abundance
sns.lineplot(data=west_df, x='date', y='green_ratio', ax=ax3, label='Greenness Ratio', color="#205a62")
ax4 = ax3.twinx()
ax4.grid(False)
# Plot insect abundance data with error bars on second subplot
ax4.errorbar(
    x=df_insect_grouped['date'],
    y=df_insect_grouped['total_mean'],
    yerr=df_insect_grouped['total_std'],
    color="#734118",
    marker='o',
    linestyle='',
    label='Insect Abundance (mean ± sd)',
)
ax3.set(
    ylim=(0, 0.9),
    # xlabel='Date', 
    ylabel='Greenness Ratio'
)
ax4.set(
    ylabel='Insect Abundance',
    ylim=(-5, 170),
)

# Third subplot: BVOC emissions with insect abundance
sns.scatterplot(
    data=df_bvoc_emp_daily,
    x='date',
    y='Total_BVOC_mean',
    ax=ax5,
    label='Emp. BVOC (mean)',
    color="#205a62",
    marker='s'
)
# ax5.errorbar(
#     x=df_bvoc_emp_daily['date'],
#     y=df_bvoc_emp_daily['Total_BVOC_mean'],
#     yerr=df_bvoc_emp_daily['Total_BVOC_std'],
#     color="#205a62",
#     marker='s',
#     linestyle='',
#     label='BVOC (mean ± sd)',
# )
ax6 = ax5.twinx()
ax6.grid(False)
# Plot insect abundance data with error bars on third subplot
ax6.errorbar(
    x=df_insect_grouped['date'],
    y=df_insect_grouped['total_mean'],
    yerr=df_insect_grouped['total_std'],
    color="#734118",
    marker='o',
    linestyle='',
    label='Insect Abundance (mean ± sd)',
)
ax5.set(
    xlabel='Date',
    ylabel=r'BVOC Emissions ng$\cdot$m$^{-2}$$\cdot$h$^{-1}$',
    ylim=(0, 1000),
    xlim=(pd.to_datetime("2021-01-01"), pd.to_datetime("2023-12-31")),
)
    
ax6.set(
    ylabel='Insect Abundance',
    ylim=(-5, 170)
)
ax6.yaxis.label.set_color("#734118")
ax2.yaxis.label.set_color("#734118")
ax4.yaxis.label.set_color("#734118")
# Combine legends from all axes and place them at the top of the figure
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
handles3, labels3 = ax3.get_legend_handles_labels()
handles5, labels5 = ax5.get_legend_handles_labels()
combined_handles = handles1 + handles2 + handles3 + handles5
combined_labels = labels1 + labels2 + labels3 + labels5

# Remove duplicate labels
unique_labels = []
unique_handles = []
for handle, label in zip(combined_handles, combined_labels):
    if label not in unique_labels:
        unique_labels.append(label)
        unique_handles.append(handle)

fig.legend(
    unique_handles, 
    unique_labels, 
    loc='upper center', 
    bbox_to_anchor=(0.51, 0.96), 
    ncol=len(unique_labels)/2, 
    frameon=True
)
ax1.text(0.01, 0.90, 'a)', transform=ax1.transAxes)
ax3.text(0.01, 0.90, 'b)', transform=ax3.transAxes)
ax5.text(0.01, 0.90, 'c)', transform=ax5.transAxes)
# Remove individual legends
ax1.get_legend().remove()
ax3.get_legend().remove()
ax5.get_legend().remove()

fig.savefig('../print/insect_impact_timeseries.png', dpi=300, bbox_inches='tight')
fig.savefig('../print/insect_impact_timeseries.pdf', dpi=300, bbox_inches='tight')
# %%
#%% Regression plots for insect abundance vs greenness metrics

# Merge insect data with CCDC data (GCC)
insect_gcc = pd.merge(
    df_insect_grouped,
    df_ccdc[['date', 'GCC']],
    on='date',
    how='inner'
)

# Merge insect data with tower camera data (greenness ratio)
# First get daily average greenness ratio
daily_green_ratio = west_df.groupby('date')['green_ratio'].mean().reset_index()
daily_green_ratio['date'] = pd.to_datetime(daily_green_ratio['date'])

insect_greenratio = pd.merge(
    df_insect_grouped,
    daily_green_ratio,
    on='date',
    how='inner'
).dropna()

# Create a figure with two subplots for regression plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Regression plot for insect abundance vs GCC
sns.regplot(
    x='GCC', 
    y='total_mean', 
    data=insect_gcc,
    scatter_kws={'alpha':0.7, 's':50},
    line_kws={'color':'red'},
    ax=ax1
)
# r_gcc, p_gcc = stats.pearsonr(insect_gcc['GCC'], insect_gcc['total_mean'])
# ax1.set_title(f'Insect Abundance vs GCC\nr = {r_gcc:.2f}, p = {p_gcc:.3f}')
ax1.set_xlabel('Green Chromatic Coordinate (GCC)')
ax1.set_ylabel('Insect Abundance (mean)')

# # Regression plot for insect abundance vs greenness ratio
sns.regplot(
    x='green_ratio', 
    y='total_mean', 
    data=insect_greenratio,
    scatter_kws={'alpha':0.7, 's':50},
    line_kws={'color':'red'},
    ax=ax2
)
# r_green, p_green = stats.pearsonr(insect_greenratio['green_ratio'], insect_greenratio['total_mean'])
# ax2.set_title(f'Insect Abundance vs Greenness Ratio\nr = {r_green:.2f}, p = {p_green:.3f}')
# ax2.set_xlabel('Greenness Ratio')
# ax2.set_ylabel('Insect Abundance (mean)')

# plt.tight_layout()
# plt.show()
# %% Regression plots for tower greeness ratio vs GCC observation
# Merge tower greenness ratio data with CCDC data (GCC)
daily_green_ratio = west_df.groupby('date')['green_ratio'].mean().reset_index()
daily_green_ratio['date'] = pd.to_datetime(daily_green_ratio['date'])
# remove green ratio = 0
daily_green_ratio = daily_green_ratio[daily_green_ratio['green_ratio'] > 0]
gcc_greenratio = pd.merge(
    daily_green_ratio,
    df_ccdc[['date', 'GCC']],
    on='date',
    how='inner'
).dropna()
gcc_greenratio['green_ratio_log'] = np.log(gcc_greenratio['green_ratio'])
# Create a figure for regression plot
fig, ax = plt.subplots(figsize=(8, 6))
# Regression plot for tower greenness ratio vs GCC
sns.regplot(
    x='green_ratio', 
    y='GCC', 
    data=gcc_greenratio,
    scatter_kws={'alpha':0.7, 's':50},
    line_kws={'color':'red'},
    ax=ax
)
r_green, p_green = stats.pearsonr(gcc_greenratio['green_ratio'], gcc_greenratio['GCC'])
ax.set_title(f'Tower Greenness Ratio vs GCC\nr = {r_green:.2f}, p = {p_green:.3f}')
ax.set_xlabel('Greenness Ratio')
ax.set_ylabel('Green Chromatic Coordinate (GCC)')
# plt.tight_layout()
# plt.show()
# fig.savefig('../print/green_ratio_vs_gcc.png', dpi=300, bbox_inches='tight')
# fig.savefig('../print/green_ratio_vs_gcc.pdf', dpi=300, bbox_inches='tight')


# %%
