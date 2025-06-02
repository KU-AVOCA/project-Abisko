'''
Tower RGB Image Analysis Script
This script analyzes tower-mounted camera images from a research site in Abisko, 
focusing on the greenness of vegetation. It processes time series data from RGB images
that have been pre-classified into understory and birch canopy components using k-means clustering.
Key features:
- Filters images by daylight hours using solar elevation calculations with pvlib
- Processes and visualizes changes in vegetation greenness over multiple years
- Separately analyzes understory and birch canopy dynamics
- Compares different view angles (North-facing vs West-facing cameras)
- Examines relationships between light conditions and greenness metrics
The script uses data from a CSV file containing:
- Image metadata (filenames, timestamps, image groups)
- K-means clustering results for two vegetation classes
- Greenness metrics (ratios, means, standard deviations, normalized values)
Visualization includes:
- Box plots showing annual variability
- Time series plots of seasonal greenness patterns by year
- Comparisons between understory and canopy greenness
- Quality control analysis of solar elevation effects

Author: Shunan Feng (shf@ign.ku.dk)
'''
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pvlib

sns.set_theme(style="darkgrid", font_scale=1.5)
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

#%% Load and process data
csvfile = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/Tower RGB images/Data_greenessByShunan_kmeans_mean/results/green_ratio_kmeans.csv'
df = pd.read_csv(csvfile)

# Rename columns for clarity
df.rename(
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
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['doys'] = df['datetime'].dt.dayofyear
df['hour'] = df['datetime'].dt.hour
df['minute'] = df['datetime'].dt.minute
df['imgroup'] = df['filename'].str.split('/').str[-3]

# Filter out night/dusk/dawn images
print("Total images before filtering:", len(df))
df['is_daytime'] = df.apply(is_daytime, axis=1)
daytime_df = df[df['is_daytime']]
# remove images taken after 2023-08-17 in west-facing camera due to overexposure
daytime_df = daytime_df[~((daytime_df['datetime'] > pd.to_datetime("2023-08-17")) & (daytime_df['imgroup'].str.contains('West')))]
print("Daytime images:", len(daytime_df))
print(f"Removed {len(df) - len(daytime_df)} images taken during night or low-light conditions")

# daytime_df = df
# # Replace values with NaN for nighttime images
# columns_to_replace = [
#     'green_ratio', 'green_mean', 'green_std', 'green_norm',
#     'understory_ratio', 'understory_mean', 'understory_std', 'understory_norm',
#     'birch_ratio', 'birch_mean', 'birch_std', 'birch_norm'
# ]

# for col in columns_to_replace:
#     if col in daytime_df.columns:
#         daytime_df.loc[~df['is_daytime'], col] = np.nan
# print("Daytime images:", len(daytime_df))
# print(f"Removed {len(df) - len(daytime_df)} images taken during night or low-light conditions")

#%% Visualization - Overall green ratio distribution by year
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=daytime_df, x='year', y='green_ratio', ax=ax)
ax.set(xlabel='Year', ylabel='Green Ratio', title='Green Ratio Distribution by Year (Daytime Images Only)')
# plt.savefig('green_ratio_by_year_daytime.png', dpi=300, bbox_inches='tight')

#%% Visualization - Green ratio by image group and year
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=daytime_df, x='year', y='green_ratio', hue='imgroup', ax=ax)
ax.set(xlabel='Year', ylabel='Green Ratio', title='Green Ratio by Image Group and Year (Daytime Images Only)')
# plt.savefig('green_ratio_by_imgroup_year_daytime.png', dpi=300, bbox_inches='tight')

#%% Visualization - Green ratio by image group and year
unique_imgroups = daytime_df['imgroup'].unique()
fig, axs = plt.subplots(len(unique_imgroups), 1, figsize=(14, 10), sharex=True)

# Create a dictionary to track all unique years across all groups
all_years = {}
colors = sns.color_palette("deep")

# First, identify all unique years and assign consistent colors
for imgroup in unique_imgroups:
    group_data = daytime_df[daytime_df['imgroup'] == imgroup]
    years = group_data['year'].unique()
    for year in years:
        if year not in all_years:
            all_years[year] = colors[len(all_years) % len(colors)]

# Plot data for each imgroup in separate subplots
for i, imgroup in enumerate(unique_imgroups):
    group_data = daytime_df[daytime_df['imgroup'] == imgroup]
    ax = axs[i]
    
    # Plot each year separately with consistent colors
    for year in sorted(group_data['year'].unique()):
        year_data = group_data[group_data['year'] == year]
        sns.lineplot(data=year_data, x='doys', y='green_ratio', ax=ax, color=all_years[year], label=year)
    
    # Remove individual legends
    if ax.get_legend():
        ax.get_legend().remove()
    
    ax.set_ylabel('Green Ratio')
    ax.set_title(f'Green Ratio Over Time - {imgroup} (Daytime Only)')
    
    # Only set xlabel for the bottom subplot
    if i == len(unique_imgroups) - 1:
        ax.set_xlabel('Day of Year')
    else:
        ax.set_xlabel('')

# Create custom legend handles
legend_elements = [Line2D([0], [0], color=color, linestyle='-', 
                          label=str(year)) 
                  for year, color in sorted(all_years.items())]

# Add a single legend for all subplots with the consistent colors
fig.legend(handles=legend_elements, title="Year", loc='upper right', bbox_to_anchor=(1.15, 0.9))
plt.tight_layout()
# plt.savefig('green_ratio_by_imgroup_daytime.png', dpi=300, bbox_inches='tight')

#%% West-facing images analysis
west_df = daytime_df[daytime_df['imgroup'].str.contains('West')]

#%% Combined plot of understory and birch ratios for west-facing images
fig, ax = plt.subplots(figsize=(14, 8))

west_df_melted = west_df.melt(id_vars=['doys', 'year'],
                             value_vars=['understory_ratio', 'birch_ratio'],
                             var_name='type', value_name='ratio_value')

sns.lineplot(data=west_df_melted, x='doys', y='ratio_value', hue='year', style='type', ax=ax, palette=sns.color_palette("deep"))

# Customize the plot
ax.set(xlabel='Day of Year',
      ylabel='Green Ratio',
      title='Comparison of Understory and Birch Green Ratios - West-facing Images (Daytime Only)')

# Create a more informative legend
legend_labels = [label.replace('understory_ratio', 'Understory').replace('birch_ratio', 'Birch') 
                for label in ax.get_legend_handles_labels()[1]]
ax.legend(ax.get_legend_handles_labels()[0], legend_labels)

plt.tight_layout()
# plt.savefig('west_understory_birch_comparison_daytime.png', dpi=300, bbox_inches='tight')

# boxplot of understory and birch ratios for west-facing images
fig, ax = plt.subplots(figsize=(14, 8))
sns.boxplot(data=west_df_melted, x='year', y='ratio_value', ax=ax, hue='type')
ax.set(xlabel='Year', ylabel='Green Ratio', title='Understory and Birch Green Ratios - West-facing Images (Daytime Only)')
plt.tight_layout()

#%% Combined plot of normalized greenness for west-facing images
fig, ax = plt.subplots(figsize=(14, 8))

west_df_melted = west_df.melt(id_vars=['doys', 'year'],
                             value_vars=['understory_norm', 'birch_norm'],
                             var_name='type', value_name='norm_value')

sns.lineplot(data=west_df_melted, x='doys', y='norm_value', hue='year', style='type', ax=ax)

# Customize the plot
ax.set(xlabel='Day of Year', 
      ylabel='Green Norm', 
      title='Comparison of Understory and Birch Green Norms - West-facing Images (Daytime Only)')
ax.set_xlim(125, 225)  # Focus on growing season
ax.set_ylim(0.35, 0.45)

# Create a more informative legend
legend_labels = [label.replace('understory_norm', 'Understory').replace('birch_norm', 'Birch') 
                for label in ax.get_legend_handles_labels()[1]]
ax.legend(ax.get_legend_handles_labels()[0], legend_labels)

plt.tight_layout()
# plt.savefig('west_norm_comparison_daytime.png', dpi=300, bbox_inches='tight')

#%% North-facing images analysis with daytime filter
north_df = daytime_df[daytime_df['imgroup'].str.contains('North')]

#%% Combined plot of understory and birch ratios for north-facing images
fig, ax = plt.subplots(figsize=(14, 8))

north_df_melted = north_df.melt(id_vars=['doys', 'year'],
                             value_vars=['understory_ratio', 'birch_ratio'],
                             var_name='type', value_name='ratio_value')

sns.lineplot(data=north_df_melted, x='doys', y='ratio_value', hue='year', style='type', ax=ax, palette=sns.color_palette("deep"))

# Customize the plot
ax.set(xlabel='Day of Year',
      ylabel='Green Ratio',
      title='Comparison of Understory and Birch Green Ratios - North-facing Images (Daytime Only)')

# Create a more informative legend
legend_labels = [label.replace('understory_ratio', 'Understory').replace('birch_ratio', 'Birch') 
                for label in ax.get_legend_handles_labels()[1]]
ax.legend(ax.get_legend_handles_labels()[0], legend_labels)

plt.tight_layout()
# plt.savefig('north_understory_birch_comparison_daytime.png', dpi=300, bbox_inches='tight')

# boxplot of understory and birch ratios for north-facing images
fig, ax = plt.subplots(figsize=(14, 8))
sns.boxplot(data=north_df_melted, x='year', y='ratio_value', ax=ax, hue='type')
ax.set(xlabel='Year', ylabel='Green Ratio', title='Understory and Birch Green Ratios - North-facing Images (Daytime Only)')
plt.tight_layout()

#%% Plot understory and birch norms for North-facing images
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot understory norm in top subplot
sns.lineplot(data=north_df, x='doys', y='understory_norm', hue='year', ax=axs[0])
axs[0].set(ylabel='Understory Norm', title='Understory Green Norm - North-facing Images (Daytime Only)')
axs[0].legend(title='Year')
axs[0].set_xlim(125, 225)  # Focus on growing season
axs[0].set_ylim(0.3, 0.5)

# Plot birch norm in bottom subplot
sns.lineplot(data=north_df, x='doys', y='birch_norm', hue='year', ax=axs[1])
axs[1].set(xlabel='Day of Year', ylabel='Birch Norm', title='Birch Green Norm - North-facing Images (Daytime Only)')
axs[1].legend(title='Year')
axs[1].set_xlim(125, 225)  # Focus on growing season
axs[1].set_ylim(0.3, 0.5)

plt.tight_layout()
# plt.savefig('north_norm_comparison_daytime.png', dpi=300, bbox_inches='tight')

#%% Add solar elevation angle to dataset for quality checking
def get_solar_elevation(row):
    """Get solar elevation angle for visualization"""
    try:
        timestamp = row['datetime']
        if pd.isna(timestamp):
            return np.nan
        
        solpos = pvlib.solarposition.get_solarposition(
            timestamp, SITE_LATITUDE, SITE_LONGITUDE, altitude=SITE_ELEVATION
        )
        return solpos['elevation'].iloc[0]
    except:
        return np.nan

# Add solar elevation to a sample of data (for debugging/visualization)
sample_df = df.sample(min(1000, len(df)))
sample_df['solar_elevation'] = sample_df.apply(get_solar_elevation, axis=1)

# Plot relationship between solar elevation and green ratio
plt.figure(figsize=(12, 6))
sns.scatterplot(data=sample_df, x='solar_elevation', y='green_ratio', hue='year')
plt.title('Green Ratio vs Solar Elevation')
plt.xlabel('Solar Elevation (degrees)')
plt.ylabel('Green Ratio')
plt.axvline(x=5, color='red', linestyle='--', label='Filtering Threshold')
plt.legend(title='Year')
plt.tight_layout()
# plt.savefig('solar_elevation_green_ratio.png', dpi=300, bbox_inches='tight')

# Plot relationship between solar elevation and green norm
plt.figure(figsize=(12, 6))
sns.scatterplot(data=sample_df, x='solar_elevation', y='green_norm', hue='year')
plt.title('Green Norm vs Solar Elevation')
plt.xlabel('Solar Elevation (degrees)')
plt.ylabel('Green Norm')
plt.axvline(x=5, color='red', linestyle='--', label='Filtering Threshold')
plt.legend(title='Year')
plt.tight_layout()
# plt.savefig('solar_elevation_green_norm.png', dpi=300, bbox_inches='tight')
# %%
