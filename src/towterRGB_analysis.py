#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pvlib
# from datetime import datetime, timedelta

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
csvfile = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/1_Simon/1_Abisko/6_Tower_Data/Tower RGB images/1_Data_greenessByShunan_kmeans/results/green_ratio_kmeans.csv'
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
df['doys'] = df['datetime'].dt.dayofyear
df['hour'] = df['datetime'].dt.hour
df['minute'] = df['datetime'].dt.minute
df['imgroup'] = df['filename'].str.split('/').str[-3]

# Filter out night/dusk/dawn images
print("Total images before filtering:", len(df))
df['is_daytime'] = df.apply(is_daytime, axis=1)
daytime_df = df[df['is_daytime']]
print("Daytime images:", len(daytime_df))
print(f"Removed {len(df) - len(daytime_df)} images taken during night or low-light conditions")

#%% Visualization - Overall green ratio distribution by year
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=daytime_df, x='year', y='green_ratio', ax=ax)
ax.set(xlabel='Year', ylabel='Green Ratio', title='Green Ratio Distribution by Year (Daytime Images Only)')
plt.savefig('green_ratio_by_year_daytime.png', dpi=300, bbox_inches='tight')

#%% Visualization - Green ratio by image group and year
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=daytime_df, x='year', y='green_ratio', hue='imgroup', ax=ax)
ax.set(xlabel='Year', ylabel='Green Ratio', title='Green Ratio by Image Group and Year (Daytime Images Only)')
plt.savefig('green_ratio_by_imgroup_year_daytime.png', dpi=300, bbox_inches='tight')
#%% Visualization - Green ratio by image group and year
unique_imgroups = daytime_df['imgroup'].unique()
fig, axs = plt.subplots(len(unique_imgroups), 1, figsize=(14, 10), sharex=True)

# Plot data for each imgroup in separate subplots
handles, labels = None, None

for i, imgroup in enumerate(unique_imgroups):
    group_data = daytime_df[daytime_df['imgroup'] == imgroup]
    g = sns.lineplot(data=group_data, x='doys', y='green_ratio', hue='year', ax=axs[i])
    
    # Store handles and labels from the first plot to use for the shared legend
    if i == 0:
        handles, labels = axs[i].get_legend_handles_labels()
    
    # Remove individual legends
    axs[i].get_legend().remove()
    
    axs[i].set_ylabel('Green Ratio')
    axs[i].set_title(f'Green Ratio Over Time - {imgroup} (Daytime Only)')
    
    # Only set xlabel for the bottom subplot
    if i == len(unique_imgroups) - 1:
        axs[i].set_xlabel('Day of Year')
    else:
        axs[i].set_xlabel('')

# Add a single legend for all subplots
fig.legend(handles, labels, title="Year", loc='upper right', bbox_to_anchor=(1.15, 0.9))
plt.tight_layout()
plt.savefig('green_ratio_by_imgroup_daytime.png', dpi=300, bbox_inches='tight')

#%% West-facing images analysis
west_df = daytime_df[daytime_df['imgroup'].str.contains('West')]

#%% Combined plot of understory and birch ratios for west-facing images
fig, ax = plt.subplots(figsize=(14, 8))

west_df_melted = west_df.melt(id_vars=['doys', 'year'],
                             value_vars=['understory_ratio', 'birch_ratio'],
                             var_name='type', value_name='ratio_value')

sns.lineplot(data=west_df_melted, x='doys', y='ratio_value', hue='year', style='type', ax=ax)

# Customize the plot
ax.set(xlabel='Day of Year',
      ylabel='Green Ratio',
      title='Comparison of Understory and Birch Green Ratios - West-facing Images (Daytime Only)')

# Create a more informative legend
legend_labels = [label.replace('understory_ratio', 'Understory').replace('birch_ratio', 'Birch') 
                for label in ax.get_legend_handles_labels()[1]]
ax.legend(ax.get_legend_handles_labels()[0], legend_labels)

plt.tight_layout()
plt.savefig('west_understory_birch_comparison_daytime.png', dpi=300, bbox_inches='tight')

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
plt.savefig('west_norm_comparison_daytime.png', dpi=300, bbox_inches='tight')

#%% North-facing images analysis with daytime filter
north_df = daytime_df[daytime_df['imgroup'].str.contains('North')]

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
plt.savefig('north_norm_comparison_daytime.png', dpi=300, bbox_inches='tight')

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

# Plot relationship between solar elevation and green metrics
fig, axs = plt.subplots(2, 1, figsize=(12, 10))
sns.scatterplot(data=sample_df, x='solar_elevation', y='green_ratio', hue='year', ax=axs[0])
axs[0].set(title='Green Ratio vs Solar Elevation', xlabel='Solar Elevation (degrees)')
axs[0].axvline(x=5, color='red', linestyle='--', label='Filtering Threshold')

sns.scatterplot(data=sample_df, x='solar_elevation', y='green_norm', hue='year', ax=axs[1])
axs[1].set(title='Green Norm vs Solar Elevation', xlabel='Solar Elevation (degrees)', 
          ylabel='Green Norm')
axs[1].axvline(x=5, color='red', linestyle='--', label='Filtering Threshold')

plt.tight_layout()
plt.savefig('solar_elevation_effect.png', dpi=300, bbox_inches='tight')
# %%
