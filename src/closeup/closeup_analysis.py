'''
Closeup Green Ratio Analysis Script

This script analyzes green ratio data derived from closeup images taken in Abisko.
It processes image data from a CSV file containing green ratio measurements of vegetation
from different locations (groups) over different years.

The script:
1. Loads green ratio data from CSV file
2. Extracts datetime, year, day of year, and group information from image filenames
3. Creates time series plots comparing green ratio across different groups for 2022 and 2023
4. Generates a boxplot comparing green ratio by day of year across different years

Input:
    - CSV file with green ratio data derived from closeup images
    - Expected filename format: "DD-MM-YYYY_GROUP.JPG" (e.g., "07-07-2022_E2.JPG")
    
Output:
    - Time series plots showing green ratio trends for different groups in 2022 and 2023
    - Boxplot comparing green ratio by day of year across years

Shunan Feng (shf@ign.ku.dk)
'''
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", font_scale=1.5)
#%% Load and process data
csvfile = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/closeup_green_ratio_stdmean/green_ratio.csv'
df = pd.read_csv(csvfile)
df['imname'] = df.filename.str.split('/').str[-1]
# Replace spaces with underscores in image names
df['imname'] = df['imname'].str.replace(' ', '_')
# Extract datetime from filenames with pattern like "07-07-2022_E2.JPG"
df['datetime'] = pd.to_datetime(df.imname.str.split('_').str[0], format='%d-%m-%Y')
df['year'] = df.datetime.dt.year
df['doy'] = df.datetime.dt.dayofyear
# Extract image group from filenames with pattern like "07-07-2022_E2.JPG", the group is E2
df['group'] = df.imname.str.split('_').str[1].str.split('.').str[0]
# %%
# Filter data by year
df_2022 = df[df['year'] == 2022]
df_2023 = df[df['year'] == 2023]

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True, sharex=True)

# 2022 plot
sns.lineplot(data=df_2022, x='doy', y='green_ratio', hue='group', ax=ax1)
sns.scatterplot(data=df_2022, x='doy', y='green_ratio', hue='group', ax=ax1, alpha=0.7, legend=False)
ax1.set_title('2022')
ax1.legend_.remove()  # Remove legend from first plot

# 2023 plot
sns.lineplot(data=df_2023, x='doy', y='green_ratio', hue='group', ax=ax2)
sns.scatterplot(data=df_2023, x='doy', y='green_ratio', hue='group', ax=ax2, alpha=0.7, legend=False)
ax2.set_title('2023')

# Move the legend outside of both plots
handles, labels = ax2.get_legend_handles_labels()
ax2.legend_.remove()  # Remove legend from second plot
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=6)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Make room for the legend

# %%
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df, x='doy', y='green_ratio', hue='year', ax=ax, palette='tab10')
ax.tick_params(bottom=True, left=True)
# %%
