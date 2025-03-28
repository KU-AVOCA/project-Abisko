#%%
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

sns.set_theme(style="darkgrid", font_scale=1.5)
#%%
csv_path = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/3_Shunan/data/studentdebug/23_06_08_orthomosaic_georef_processed_resampled_nearest/NDVI"
# Get all CSV files in the directory
csv_files = glob.glob(os.path.join(csv_path, "*.csv"))

# Initialize an empty list to store dataframes
dfs = []

# Read each CSV file and add a column for the file name
for file in csv_files:
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Extract just the filename without path
    filename = os.path.basename(file)
    
    # Add the filename as a new column
    df['imgid'] = filename
    
    # Append to the list
    dfs.append(df)

# Combine all dataframes
df = pd.concat(dfs, ignore_index=True)
# %%
sns.lineplot(data=df, x='resolution', y='glcm_homogeneity', marker='o', markersize=8)
#%%
sns.lineplot(data=df, x='resolution', y='glcm_contrast', marker='s', markersize=8)
#%%
sns.lineplot(data=df, x='resolution', y='glcm_dissimilarity', marker='^', markersize=8)
# %%
# sns.lineplot(data=df, x='resolution', y='mean')
# %%
