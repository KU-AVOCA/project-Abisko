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
import scipy.stats as stats

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
csvfile = "/data_3/shunan_2/KU/Data_greenes_thermal_kmeans_mean/results/green_ratio_thermal_kmeans.csv"
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
        'class2_norm': 'birch_norm',

        'class1_temp_mean': 'understory_temp_mean',
        'class1_temp_std': 'understory_temp_std',
        'class2_temp_mean': 'birch_temp_mean',
        'class2_temp_std': 'birch_temp_std'
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



#%% Visualization - Overall green ratio distribution by year
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=daytime_df, x='year', y='green_ratio', ax=ax)
ax.set(xlabel='Year', ylabel='Green Ratio', title='Green Ratio Distribution by Year (Daytime Images Only)')
# plt.savefig('green_ratio_by_year_daytime.png', dpi=300, bbox_inches='tight')

#%% Visualization - Overall temperature distribution by year
fig, ax = plt.subplots(figsize=(10, 6))
temp_df = daytime_df[['year', 'understory_temp_mean', 'birch_temp_mean']].melt(
    id_vars='year', var_name='type', value_name='temp'
)
temp_df['type'] = temp_df['type'].map({
    'understory_temp_mean': 'Understory Temp',
    'birch_temp_mean': 'Birch Temp'
})
sns.boxplot(data=temp_df, x='year', y='temp', hue='type', ax=ax, notch=True)
ax.set(xlabel='Year', ylabel='Temperature (°C)', title='Temperature by Year (Daytime Images Only)')
ax.legend(title='')
plt.tight_layout()


#%% Visualization - time series of green ratio, temperature for understory and birch
fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Prepare year palette
years_sorted = sorted(daytime_df['year'].dropna().unique())
palette = dict(zip(years_sorted, sns.color_palette("deep", n_colors=len(years_sorted))))

# --- Green ratio (both Understory and Birch) ---
ratio_melted = daytime_df.melt(
    id_vars=['doys', 'year'],
    value_vars=['understory_ratio', 'birch_ratio'],
    var_name='type',
    value_name='ratio_value'
)
ratio_melted['type'] = ratio_melted['type'].map({'understory_ratio': 'Understory', 'birch_ratio': 'Birch'})

sns.lineplot(
    data=ratio_melted,
    x='doys',
    y='ratio_value',
    hue='year',
    style='type',
    ax=axs[0],
    palette=palette,
    dashes=True
)
axs[0].set(ylabel='Green Ratio', title='Understory and Birch Green Ratio Over Time (Daytime Images Only)')

# --- Temperature (both Understory and Birch) ---
temp_melted = daytime_df.melt(
    id_vars=['doys', 'year'],
    value_vars=['understory_temp_mean', 'birch_temp_mean'],
    var_name='type',
    value_name='temp_value'
)
temp_melted['type'] = temp_melted['type'].map({'understory_temp_mean': 'Understory', 'birch_temp_mean': 'Birch'})

sns.lineplot(
    data=temp_melted,
    x='doys',
    y='temp_value',
    hue='year',
    style='type',
    ax=axs[1],
    palette=palette,
    dashes=True
)
axs[1].set(xlabel='Day of Year', ylabel='Temperature (°C)', title='Understory and Birch Temperature Over Time (Daytime Images Only)')

# Remove axes legends and add a single combined legend (years = colors, types = linestyles)
for ax in axs:
    if ax.get_legend() is not None:
        ax.get_legend().remove()

# Year handles (colors)
year_handles = [Line2D([0], [0], color=palette[y], lw=2, label=str(y)) for y in years_sorted]
# Type handles (linestyles)
type_handles = [
    Line2D([0], [0], color='k', lw=2, linestyle='-', label='Understory'),
    Line2D([0], [0], color='k', lw=2, linestyle='-.', label='Birch')
]

fig.legend(handles=year_handles + type_handles, loc='upper right', title='Year / Type', bbox_to_anchor=(1.15, 0.9))
plt.tight_layout()
# plt.savefig('understory_birch_ratio_temp_timeseries_daytime.png', dpi=300, bbox_inches='tight')

#%% statistical test to compare the temperature between understory and birch in different years

for year in sorted(daytime_df['year'].dropna().unique()):
    understory_temps = daytime_df[daytime_df['year'] == year]['understory_temp_mean'].dropna()
    birch_temps = daytime_df[daytime_df['year'] == year]['birch_temp_mean'].dropna()
    t_stat, p_value = stats.ttest_ind(understory_temps, birch_temps, equal_var=False)
    print(f"Year {year}: t-statistic = {t_stat:.3f}, p-value = {p_value:.3e}")
    if p_value < 0.05:
        print(f"  -> Significant difference in temperatures between Understory and Birch (p < 0.05)")
        print(f"     Understory mean = {understory_temps.mean():.2f}, Birch mean = {birch_temps.mean():.2f}")
        print(f"     Understory std = {understory_temps.std():.2f}, Birch std = {birch_temps.std():.2f}")
    else:
        print(f"  -> No significant difference in temperatures between Understory and Birch (p >= 0.05)")


#%% convert data to daily averages
daytime_df = daytime_df.groupby(['imgroup', 'year', 'doys']).agg({
    'green_ratio': 'mean',
    'understory_ratio': 'mean',
    'birch_ratio': 'mean',
    'understory_norm': 'mean',
    'birch_norm': 'mean',
    'understory_temp_mean': 'mean',
    'birch_temp_mean': 'mean'
}).reset_index()

#%% statistical test to compare daily temperature between understory and birch in different years
for year in sorted(daytime_df['year'].dropna().unique()):
    understory_temps = daytime_df[daytime_df['year'] == year]['understory_temp_mean'].dropna()
    birch_temps = daytime_df[daytime_df['year'] == year]['birch_temp_mean'].dropna()
    t_stat, p_value = stats.ttest_ind(understory_temps, birch_temps, equal_var=False)
    print(f"(Daily Avg) Year {year}: t-statistic = {t_stat:.3f}, p-value = {p_value:.3e}")
    if p_value < 0.05:
        print(f"  -> Significant difference in daily avg temperatures between Understory and Birch (p < 0.05)")
        print(f"     Understory mean = {understory_temps.mean():.2f}, Birch mean = {birch_temps.mean():.2f}")
        print(f"     Understory std = {understory_temps.std():.2f}, Birch std = {birch_temps.std():.2f}")
    else:
        print(f"  -> No significant difference in daily avg temperatures between Understory and Birch (p >= 0.05)")
# %%
