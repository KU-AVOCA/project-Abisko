#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmocean as cm
import pandas as pd
import os
import pvlib

sns.set_theme(style="darkgrid", font_scale=1.5)

#%% Define Abisko coordinates (same as RGB analysis)
SITE_LATITUDE = 68.34808742 # in decimal degrees 
SITE_LONGITUDE = 19.05077561 # in decimal degrees
SITE_ELEVATION = 400  # meters above sea level (approximate)

#%% Function to determine daylight hours (same as RGB analysis)
def is_daytime(row, min_elevation=5.0):
    """
    Determine if a timestamp is during daylight hours based on solar elevation.
    """
    try:
        timestamp = row['datetime'] if 'datetime' in row else row['time']
        if pd.isna(timestamp):
            return False
        
        solpos = pvlib.solarposition.get_solarposition(
            timestamp, 
            SITE_LATITUDE, 
            SITE_LONGITUDE,
            altitude=SITE_ELEVATION
        )
        
        elevation = solpos['elevation'].iloc[0]
        return elevation > min_elevation
        
    except Exception as e:
        print(f"Error calculating solar position: {e}")
        return False

# %% Load thermal data
thermal_csv = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/Tower thermal images/preview/north/thermal_stats.csv'
df_thermal = pd.read_csv(thermal_csv)
df_thermal['time'] = pd.to_datetime(df_thermal['time'], format='%Y-%m-%d_%H.%M.%S')
df_thermal = df_thermal.sort_values('time').reset_index(drop=True)
df_thermal['year'] = df_thermal['time'].dt.year
df_thermal['month'] = df_thermal['time'].dt.month
df_thermal['doys'] = df_thermal['time'].dt.dayofyear
df_thermal['hour'] = df_thermal['time'].dt.hour
df_thermal['minute'] = df_thermal['time'].dt.minute
df_thermal['datetime'] = df_thermal['time']  # For compatibility with is_daytime function

# Filter thermal data for daylight hours
print("Total thermal images before filtering:", len(df_thermal))
df_thermal['is_daytime'] = df_thermal.apply(is_daytime, axis=1)
daytime_thermal = df_thermal[df_thermal['is_daytime']]
print("Daytime thermal images:", len(daytime_thermal))

#%% Load RGB data (west-facing since thermal "north" is actually west-facing)
rgb_csv = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/Tower RGB images/Data_greenessByShunan_kmeans_mean/results/green_ratio_kmeans.csv'
df_rgb = pd.read_csv(rgb_csv)

# Process RGB data (same as towterRGB_analysis.py)
df_rgb.rename(
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

df_rgb['datetime'] = pd.to_datetime(df_rgb['datetime'], errors='coerce')
df_rgb['year'] = df_rgb['datetime'].dt.year
df_rgb['month'] = df_rgb['datetime'].dt.month
df_rgb['doys'] = df_rgb['datetime'].dt.dayofyear
df_rgb['hour'] = df_rgb['datetime'].dt.hour
df_rgb['minute'] = df_rgb['datetime'].dt.minute
df_rgb['imgroup'] = df_rgb['filename'].str.split('/').str[-3]

# Filter RGB data for daylight hours and west-facing (to match thermal)
print("Total RGB images before filtering:", len(df_rgb))
df_rgb['is_daytime'] = df_rgb.apply(is_daytime, axis=1)
daytime_rgb = df_rgb[df_rgb['is_daytime']]
# Remove overexposed images after 2023-08-17 in west-facing camera
daytime_rgb = daytime_rgb[~((daytime_rgb['datetime'] > pd.to_datetime("2023-08-17")) & (daytime_rgb['imgroup'].str.contains('West')))]
# Filter for west-facing images (since thermal "north" is actually west)
west_rgb = daytime_rgb[daytime_rgb['imgroup'].str.contains('West')]
print("Daytime west-facing RGB images:", len(west_rgb))

#%% Create consistent color palette (same as RGB analysis)
unique_years = sorted(list(set(daytime_thermal['year'].unique()).union(set(west_rgb['year'].unique()))))
colors = sns.color_palette("deep")
year_colors = {year: colors[i % len(colors)] for i, year in enumerate(unique_years)}

#%% Combined time series plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# Plot thermal temperature time series
for year in sorted(daytime_thermal['year'].unique()):
    year_data = daytime_thermal[daytime_thermal['year'] == year]
    sns.lineplot(data=year_data, x='doys', y='mean_temp', ax=ax1, 
                color=year_colors[year], label=f'{year}', alpha=0.8)

ax1.set_ylabel('Mean Temperature (°C)')
ax1.set_title('Tower Thermal Camera - Mean Temperature (West-facing, Daytime Only)')
ax1.legend(title='Year', loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot RGB green ratio time series
for year in sorted(west_rgb['year'].unique()):
    year_data = west_rgb[west_rgb['year'] == year]
    sns.lineplot(data=year_data, x='doys', y='green_ratio', ax=ax2, 
                color=year_colors[year], label=f'{year}', alpha=0.8)

ax2.set_ylabel('Green Ratio')
ax2.set_xlabel('Day of Year')
ax2.set_title('Tower RGB Camera - Green Ratio (West-facing, Daytime Only)')
ax2.legend(title='Year', loc='upper right')
ax2.grid(True, alpha=0.3)

# Set consistent x-axis limits
ax1.set_xlim(100, 300)  # Focus on growing season
ax2.set_xlim(100, 300)

plt.tight_layout()
plt.show()

#%% Alternative: Dual-axis plot (temperature and green ratio on same plot)
fig, ax1 = plt.subplots(figsize=(16, 8))

# Plot thermal data
for year in sorted(daytime_thermal['year'].unique()):
    year_data = daytime_thermal[daytime_thermal['year'] == year]
    sns.lineplot(data=year_data, x='doys', y='mean_temp', ax=ax1, 
                color=year_colors[year], linestyle='-', label=f'Temperature {year}')

ax1.set_xlabel('Day of Year')
ax1.set_ylabel('Mean Temperature (°C)', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_title('Tower Camera Data - Temperature and Green Ratio (West-facing, Daytime Only)')

# Create second y-axis for green ratio
ax2 = ax1.twinx()
for year in sorted(west_rgb['year'].unique()):
    year_data = west_rgb[west_rgb['year'] == year]
    sns.lineplot(data=year_data, x='doys', y='green_ratio', ax=ax2, 
                color=year_colors[year], linestyle='--', label=f'Green Ratio {year}')

ax2.set_ylabel('Green Ratio', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))

ax1.set_xlim(100, 300)
ax1.set_ylim(-15, 25)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%% Daily aggregated comparison
# Aggregate thermal data to daily means
thermal_daily = daytime_thermal.groupby(['year', 'doys']).agg({
    'mean_temp': 'mean',
    'min_temp': 'mean',
    'max_temp': 'mean'
}).reset_index()

# Aggregate RGB data to daily means  
rgb_daily = west_rgb.groupby(['year', 'doys']).agg({
    'green_ratio': 'mean',
    'understory_ratio': 'mean',
    'birch_ratio': 'mean'
}).reset_index()

# Combined daily plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# Daily thermal
for year in sorted(thermal_daily['year'].unique()):
    year_data = thermal_daily[thermal_daily['year'] == year]
    ax1.plot(year_data['doys'], year_data['mean_temp'], 
            color=year_colors[year], label=f'{year}', linewidth=2)

ax1.set_ylabel('Daily Mean Temperature (°C)')
ax1.set_title('Daily Aggregated Tower Camera Data (West-facing, Daytime Only)')
ax1.legend(title='Year')
ax1.grid(True, alpha=0.3)

# Daily RGB
for year in sorted(rgb_daily['year'].unique()):
    year_data = rgb_daily[rgb_daily['year'] == year]
    ax2.plot(year_data['doys'], year_data['green_ratio'], 
            color=year_colors[year], label=f'{year}', linewidth=2)

ax2.set_ylabel('Daily Mean Green Ratio')
ax2.set_xlabel('Day of Year')
ax2.legend(title='Year')
ax2.grid(True, alpha=0.3)

ax1.set_xlim(100, 300)
ax1.set_ylim(-15, 25)
ax2.set_xlim(100, 300)

plt.tight_layout()
plt.show()

# %% Summary statistics
print("\n=== Thermal Data Summary ===")
print(f"Years available: {sorted(daytime_thermal['year'].unique())}")
print(f"Temperature range: {daytime_thermal['mean_temp'].min():.1f}°C to {daytime_thermal['mean_temp'].max():.1f}°C")
print(f"Date range: {daytime_thermal['time'].min()} to {daytime_thermal['time'].max()}")

print("\n=== RGB Data Summary ===")
print(f"Years available: {sorted(west_rgb['year'].unique())}")
print(f"Green ratio range: {west_rgb['green_ratio'].min():.3f} to {west_rgb['green_ratio'].max():.3f}")
print(f"Date range: {west_rgb['datetime'].min()} to {west_rgb['datetime'].max()}")