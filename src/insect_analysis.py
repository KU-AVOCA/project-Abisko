'''
Insect Analysis Script for Abisko Project
This script analyzes data from multiple sources to examine relationships between
vegetation greenness, insect abundance, and BVOC (Biogenic Volatile Organic Compounds)
emissions in Abisko, Sweden. The script performs the following functions:
1. Determines daylight hours based on solar elevation angles using pvlib
2. Processes and filters tower camera images based on daylight conditions
3. Loads and analyzes insect abundance survey data
4. Loads and analyzes BVOC emission data 
5. Creates visualizations showing relationships between:
    - Greenness ratio from west-facing camera images and insect abundance
    - Greenness ratio and BVOC emissions (per species and combined)
    - Greenness ratio from tower images vs close-up images
6. Performs statistical analysis of correlations between greenness and BVOC emissions
The script handles data across multiple years (2022-2023) and provides visualizations
for comparing seasonal patterns in vegetation phenology, insect populations, and
plant volatile emissions.


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

#%% read insect abundance data
df_insect = pd.read_excel('/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/Insect_Survey/Insect_Surveys_2022-2023_Abisko.xlsx', sheet_name='all')
df_insect['date'] = pd.to_datetime(df_insect['date'], errors='coerce')
df_insect['year'] = df_insect['date'].dt.year
df_insect['month'] = df_insect['date'].dt.month
df_insect['doys'] = df_insect['date'].dt.dayofyear
sns.barplot(data=df_insect, x='doys', y='total', hue='year', errorbar='sd')


#%% read bvoc data
df_bvoc = pd.read_csv('/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/BVOC/Data_Master_Abisko_2022-2023_V4.csv',
                      delimiter=';')
df_bvoc['date'] = pd.to_datetime(df_bvoc['Date'], errors='coerce')
df_bvoc['year'] = df_bvoc['date'].dt.year
df_bvoc['month'] = df_bvoc['date'].dt.month
df_bvoc['doys'] = df_bvoc['date'].dt.dayofyear

sns.lineplot(data=df_bvoc, x='doys', y='Total_BVOC', hue='year', errorbar='sd', style='Species')

#%% West-facing images analysis
west_df = daytime_df[daytime_df['imgroup'].str.contains('West')]

#%% plot the general greenness ratio for west-facing images with insect abundance
fig, ax1 = plt.subplots(figsize=(14, 8))

# Define a consistent color palette for years
years = sorted(list(set(west_df['year'].unique()).union(set(df_insect['year'].unique()))))
wigglytuff_palette =["#205a62", "#52a48b", "#734118"]
year_palette = {year: color for year, color in zip(years, wigglytuff_palette)}

# Plot greenness data
sns.lineplot(data=west_df, x='doys', y='green_ratio', hue='year', ax=ax1, palette=year_palette)

# Create second y-axis
ax2 = ax1.twinx()
ax2.grid(False)

# Plot insect abundance with consistent colors
for year in df_insect['year'].unique():
    year_data = df_insect[df_insect['year'] == year]
    # Group by doys and calculate mean/std for error bars
    year_data_grouped = year_data.groupby('doys').agg({
        'total': ['mean', 'std']
    }).reset_index()
    year_data_grouped.columns = ['doys', 'total', 'sd']  # Flatten column names

    # Use the grouped data for plotting with error bars
    for _, row in year_data_grouped.iterrows():
        ax2.errorbar(
            x=row['doys'],
            y=row['total'],
            yerr=row['sd'],
            color=year_palette[year],
            marker='o',
            linestyle='',
            label=f"Insects {year}" if _ == 0 else ""  # Only add label for first point
        )

# Set axis labels and limits
ax1.set_xlabel('Day of Year')
ax1.set_ylabel('Green Ratio')
ax2.set_ylabel('Insect Abundance')
ax1.set_xlim(125, 300)
ax1.set_ylim(-0.01, 0.9)  # Adjusted to focus on the growing season

# Align zero points on both axes
y1_min, y1_max = ax1.get_ylim()
y2_min, y2_max = ax2.get_ylim()

# Calculate the position of 0 in normalized coordinates for both axes
if y1_min < 0 < y1_max:
    zero_pos_y1 = -y1_min / (y1_max - y1_min)
else:
    zero_pos_y1 = 0  # Default if 0 is not in the range

if y2_min < 0 < y2_max:
    zero_pos_y2 = -y2_min / (y2_max - y2_min)
else:
    zero_pos_y2 = 0  # Default if 0 is not in the range

# Adjust limits to align zeros
if zero_pos_y1 != zero_pos_y2:
    if y1_min >= 0 and y2_min >= 0:
        # If both ranges start above 0, just align the bottoms
        ax1.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
    else:
        # Adjust the axis with the smaller normalized zero position
        if zero_pos_y1 < zero_pos_y2:
            # Adjust y2 to match y1's zero position
            new_y2_min = -zero_pos_y1 * (y2_max - 0) / (1 - zero_pos_y1)
            ax2.set_ylim(new_y2_min, y2_max)
        else:
            # Adjust y1 to match y2's zero position
            new_y1_min = -zero_pos_y2 * (y1_max - 0) / (1 - zero_pos_y2)
            ax1.set_ylim(new_y1_min, y1_max)

# Create a combined legend
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, [f"Greenness {label}" for label in labels1] + labels2, loc='best')

# Remove the second legend
ax2.get_legend().remove() if ax2.get_legend() else None
# plt.title('Greenness Ratio and Insect Abundance - West-facing Images')
plt.tight_layout()
fig.savefig('/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/analysis/print/greenness_ratio_insect_abundance_west.png', dpi=300)

#%% plot the general greenness ratio for west-facing images with Total_BVOC
fig, ax1 = plt.subplots(figsize=(14, 8))

# Define a consistent color palette for years
years = sorted(list(set(west_df['year'].unique()).union(set(df_bvoc['year'].unique()))))
wigglytuff_palette =["#205a62", "#52a48b", "#734118"]
year_palette = {year: color for year, color in zip(years, wigglytuff_palette)}

# Plot greenness data
sns.lineplot(data=west_df, x='doys', y='green_ratio', hue='year', ax=ax1, palette=year_palette)

# Create second y-axis
ax2 = ax1.twinx()
ax2.grid(False)

# Plot BVOC data with consistent colors
for year in df_bvoc['year'].unique():
    year_data = df_bvoc[df_bvoc['year'] == year]
    # Group by species and doys
    for species, species_data in year_data.groupby('Species'):
        # Group by doys and calculate mean/std for error bars
        species_data_grouped = species_data.groupby('doys').agg({
            'Total_BVOC': ['mean', 'std']
        }).reset_index()
        species_data_grouped.columns = ['doys', 'Total_BVOC', 'sd']  # Flatten column names

        # Use the grouped data for plotting with error bars
        # Define markers for different species
        markers = {'Grass': 's', 'Emp.': '^', 'Vac.': 'o'}
        
        # Get the marker for this species (default to 'x' if not in the dictionary)
        marker = markers.get(species, 'x')
        
        for i, row in species_data_grouped.iterrows():
            ax2.errorbar(
            x=row['doys'],
            y=row['Total_BVOC'],
            yerr=row['sd'],
            color=year_palette[year],
            marker=marker,  # Use the species-specific marker
            linestyle='',
            label=f"BVOC {species} {year}" if i == 0 else ""  # Only add label for first point
            )

# Set axis labels and limits
ax1.set_xlabel('Day of Year')
ax1.set_ylabel('Green Ratio')
ax2.set_ylabel('Total BVOC Emissions')
ax1.set_xlim(125, 300)
ax1.set_ylim(-0.01, 0.9)  # Adjusted to focus on the growing season
ax2.set_ylim(0, 2000)  # Adjusted to focus on the growing season
# # Align zero points on both axes
# y1_min, y1_max = ax1.get_ylim()
# y2_min, y2_max = ax2.get_ylim()

# # Calculate the position of 0 in normalized coordinates for both axes
# if y1_min < 0 < y1_max:
#     zero_pos_y1 = -y1_min / (y1_max - y1_min)
# else:
#     zero_pos_y1 = 0  # Default if 0 is not in the range

# if y2_min < 0 < y2_max:
#     zero_pos_y2 = -y2_min / (y2_max - y2_min)
# else:
#     zero_pos_y2 = 0  # Default if 0 is not in the range

# # Adjust limits to align zeros
# if zero_pos_y1 != zero_pos_y2:
#     if y1_min >= 0 and y2_min >= 0:
#         # If both ranges start above 0, just align the bottoms
#         ax1.set_ylim(bottom=0)
#         ax2.set_ylim(bottom=0)
#     else:
#         # Adjust the axis with the smaller normalized zero position
#         if zero_pos_y1 < zero_pos_y2:
#             # Adjust y2 to match y1's zero position
#             new_y2_min = -zero_pos_y1 * (y2_max - 0) / (1 - zero_pos_y1)
#             ax2.set_ylim(new_y2_min, y2_max)
#         else:
#             # Adjust y1 to match y2's zero position
#             new_y1_min = -zero_pos_y2 * (y1_max - 0) / (1 - zero_pos_y2)
#             ax1.set_ylim(new_y1_min, y1_max)

# Create a combined legend
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, [f"Greenness {label}" for label in labels1] + labels2, loc='best')

# Remove the second legend
ax2.get_legend().remove() if ax2.get_legend() else None
plt.tight_layout()
fig.savefig('/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/analysis/print/greenness_ratio_bvoc_per_species_west.png', dpi=300)
# %%
#%% plot the general greenness ratio for west-facing images with Total_BVOC (combined across species)
fig, ax1 = plt.subplots(figsize=(14, 8))

# Define a consistent color palette for years
years = sorted(list(set(west_df['year'].unique()).union(set(df_bvoc['year'].unique()))))
wigglytuff_palette =["#205a62", "#52a48b", "#734118"]
year_palette = {year: color for year, color in zip(years, wigglytuff_palette)}

# Plot greenness data
sns.lineplot(data=west_df, x='doys', y='green_ratio', hue='year', ax=ax1, palette=year_palette)

# Create second y-axis
ax2 = ax1.twinx()
ax2.grid(False)

# Plot BVOC data with consistent colors but combined across species
for year in df_bvoc['year'].unique():
    year_data = df_bvoc[df_bvoc['year'] == year]
    
    # Group by doys (combining all species) and calculate mean/std
    combined_data = year_data.groupby('doys').agg({
        'Total_BVOC': ['mean', 'std']
    }).reset_index()
    combined_data.columns = ['doys', 'Total_BVOC', 'sd']  # Flatten column names

    # Use the grouped data for plotting with error bars
    for i, row in combined_data.iterrows():
        ax2.errorbar(
            x=row['doys'],
            y=row['Total_BVOC'],
            yerr=row['sd'],
            color=year_palette[year],
            marker='o',
            linestyle='',
            label=f"BVOC {year}" if i == 0 else ""  # Only add label for first point
        )

# Set axis labels and limits
ax1.set_xlabel('Day of Year')
ax1.set_ylabel('Green Ratio')
ax2.set_ylabel('Total BVOC Emissions')
ax1.set_xlim(125, 300)
ax1.set_ylim(-0.01, 0.9)  # Adjusted to focus on the growing season

# Align zero points on both axes
# y1_min, y1_max = ax1.get_ylim()
# y2_min, y2_max = ax2.get_ylim()

# # Calculate the position of 0 in normalized coordinates for both axes
# if y1_min < 0 < y1_max:
#     zero_pos_y1 = -y1_min / (y1_max - y1_min)
# else:
#     zero_pos_y1 = 0  # Default if 0 is not in the range

# if y2_min < 0 < y2_max:
#     zero_pos_y2 = -y2_min / (y2_max - y2_min)
# else:
#     zero_pos_y2 = 0  # Default if 0 is not in the range

# # Adjust limits to align zeros
# if zero_pos_y1 != zero_pos_y2:
#     if y1_min >= 0 and y2_min >= 0:
#         # If both ranges start above 0, just align the bottoms
#         ax1.set_ylim(bottom=0)
#         ax2.set_ylim(bottom=0)
#     else:
#         # Adjust the axis with the smaller normalized zero position
#         if zero_pos_y1 < zero_pos_y2:
#             # Adjust y2 to match y1's zero position
#             new_y2_min = -zero_pos_y1 * (y2_max - 0) / (1 - zero_pos_y1)
#             ax2.set_ylim(new_y2_min, y2_max)
#         else:
#             # Adjust y1 to match y2's zero position
#             new_y1_min = -zero_pos_y2 * (y1_max - 0) / (1 - zero_pos_y2)
#             ax1.set_ylim(new_y1_min, y1_max)

# Create a combined legend
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, [f"Greenness {label}" for label in labels1] + labels2, loc='best')

# Remove the second legend
ax2.get_legend().remove() if ax2.get_legend() else None
plt.title('Greenness Ratio and Combined BVOC Emissions - West-facing Images')
plt.tight_layout()
fig.savefig('/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/analysis/print/greenness_ratio_bvoc_all_species_west.png', dpi=300)
# %% plot the GreenRatio and Total BVOC for each species as time series (doys)
# Create a time series plot for each species
# Define a consistent color palette for years
years = sorted(list(set(west_df['year'].unique()).union(set(df_bvoc['year'].unique()))))
wigglytuff_palette =["#205a62", "#52a48b", "#734118"]
year_palette = {year: color for year, color in zip(years, wigglytuff_palette)}

fig, ax1 = plt.subplots(figsize=(14, 8))
# Plot greenness data
sns.lineplot(data=df_bvoc, 
             x='doys', y='GreenRatio', hue='year', 
             palette=year_palette, errorbar='sd', markers=True)
# Create second y-axis
ax2 = ax1.twinx()
ax2.grid(False)
# Plot BVOC data with consistent colors
for year in df_bvoc['year'].unique():
    year_data = df_bvoc[df_bvoc['year'] == year]
    combined_data = year_data.groupby('doys').agg({
        'Total_BVOC': ['mean', 'std']
    }).reset_index()
    combined_data.columns = ['doys', 'Total_BVOC', 'sd']  # Flatten column names
    # Use the grouped data for plotting with error bars
    for i, row in combined_data.iterrows():
        ax2.errorbar(
            x=row['doys'],
            y=row['Total_BVOC'],
            yerr=row['sd'],
            color=year_palette[year],
            marker='o',
            linestyle='',
            label=f"BVOC {year}" if i == 0 else ""  # Only add label for first point
        )

# Set axis labels and limits
ax1.set_xlabel('Day of Year')
ax1.set_ylabel('Green Ratio')
ax2.set_ylabel('Total BVOC Emissions')
ax1.set_xlim(125, 300)

# Create a combined legend
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, [f"Greenness {label}" for label in labels1] + labels2, loc='best')

# Remove the second legend
ax2.get_legend().remove() if ax2.get_legend() else None
plt.title('Greenness Ratio and Total BVOC Emissions - All Species')
plt.tight_layout()

# %%
sns.relplot(
    data=df_bvoc, 
    x='doys', 
    y='Total_BVOC', 
    hue='year', 
    col='Species', 
    kind='line'
    )
# %% pairplot
# Create a pairplot to visualize relationships between variables
df_bvoc['log_Total_BVOC'] = np.log(df_bvoc['Total_BVOC'] + 1)  # Adding 1 to avoid log(0)
sns.pairplot(data=df_bvoc, hue='Species', vars=['log_Total_BVOC', 'doys', 'GreenRatio'])
# %% regplot of green ratio and total BVOC for each species
# Create a regplot for each species
species_list = df_bvoc['Species'].unique()
for species in species_list:
    plt.figure(figsize=(10, 6))
    species_data = df_bvoc[df_bvoc['Species'] == species].dropna(subset=['GreenRatio', 'Total_BVOC'])
    
    # Calculate correlation coefficient and p-value
    r, p = stats.pearsonr(species_data['GreenRatio'], 
                          species_data['Total_BVOC'])
    
    # Create the regplot
    sns.regplot(data=species_data, x='GreenRatio', y='Total_BVOC')
    
    # Add r and p-value information to the plot
    plt.annotate(f'r = {r:.2f}, p = {p:.3f}', 
                 xy=(0.05, 0.95), 
                 xycoords='axes fraction',
                 fontsize=12,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    plt.title(f'Regression Plot for {species}')
    plt.xlabel('Green Ratio')
    plt.ylabel('Total BVOC Emissions')
    plt.tight_layout()
    plt.show()


#%% figure for Simon to plot time series of Green ratio from tower images and close ups

df_closeup = pd.read_csv('/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/closeup_green_ratio_mean/green_ratio.csv')
df_closeup['imname'] = df_closeup.filename.str.split('/').str[-1]
# Replace spaces with underscores in image names
df_closeup['imname'] = df_closeup['imname'].str.replace(' ', '_')
# Extract datetime from filenames with pattern like "07-07-2022_E2.JPG"
df_closeup['datetime'] = pd.to_datetime(df_closeup.imname.str.split('_').str[0], format='%d-%m-%Y')
df_closeup['year'] = df_closeup.datetime.dt.year
df_closeup['doy'] = df_closeup.datetime.dt.dayofyear
# Extract image group from filenames with pattern like "07-07-2022_E2.JPG", the group is E2
df_closeup['group'] = df_closeup.imname.str.split('_').str[1].str.split('.').str[0]
# plot the general greenness ratio for west-facing images with green ratio from close up images
fig, ax1 = plt.subplots(figsize=(14, 8))
# Define a consistent color palette for years
years = sorted(list(set(west_df['year'].unique()).union(set(df_closeup['year'].unique()))))
wigglytuff_palette =["#205a62", "#52a48b", "#734118"]
year_palette = {year: color for year, color in zip(years, wigglytuff_palette)}
# Plot greenness data
sns.lineplot(data=west_df, x='doys', y='green_ratio', hue='year', ax=ax1, palette=year_palette)
# Create second y-axis
ax2 = ax1.twinx()
ax2.grid(False)
# Plot closeup data with consistent colors
for year in df_closeup['year'].unique():
    year_data = df_closeup[df_closeup['year'] == year]
    # Group by doys and calculate mean/std for error bars
    year_data_grouped = year_data.groupby('doy').agg({
        'green_ratio': ['mean', 'std']
    }).reset_index()
    year_data_grouped.columns = ['doy', 'green_ratio', 'sd']  # Flatten column names
    # Use the grouped data for plotting with error bars
    for i, row in year_data_grouped.iterrows():
        ax2.errorbar(
            x=row['doy'],
            y=row['green_ratio'],
            yerr=row['sd'],
            color=year_palette[year],
            marker='o',
            linestyle='',
            label=f"Closeup {year}" if i == 0 else ""  # Only add label for first point
        )

# Set axis labels and limits
ax1.set_xlabel('Day of Year')
ax1.set_ylabel('Green Ratio from Tower Images')
ax2.set_ylabel('Green Ratio from Closeup Images')
ax1.set_xlim(125, 300)
ax1.set_ylim(-0.01, 0.9)  # Adjusted to focus on the growing season
ax2.set_ylim(-0.01, 0.9)  # same as tower images

# Create a combined legend
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, [f"Greenness {label}" for label in labels1] + labels2, loc='best')
# Remove the second legend
ax2.get_legend().remove() if ax2.get_legend() else None
# plt.title('Greenness Ratio from Tower Images and Closeup Images')
plt.tight_layout()
fig.savefig('/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/analysis/print/greenness_ratio_closeup4Simon.png', dpi=300)

# %%
