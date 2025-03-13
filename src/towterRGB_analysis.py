#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", font_scale=1.5)

# %%
csvfile = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/1_Personal_folders/1_Simon/1_Abisko/6_Tower_Data/Tower RGB images/1_Data_greenessByShunan_kmeans/results/green_ratio_kmeans.csv'
df = pd.read_csv(csvfile)
# %%
# replace column names with more meaningful names
# class1_ratio -> understory_ratio, class1_mean -> understory_mean, class1_std -> understory_std, class1_norm -> understory_norm
# class2_ratio -> birch_ratio, class2_mean -> birch_mean, class2_std -> birch_std, class2_norm -> birch_norm
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
#%%
df['datetime'] = pd.to_datetime(df['datetime'])
df['year'] = df['datetime'].dt.year
df['doys'] = df['datetime'].dt.dayofyear
df['imgroup'] = df['filename'].str.split('/').str[-3]
# %%
# fig, ax = plt.subplots(figsize=(12, 6))
# sns.lineplot(data=df, x='doys', y='green_ratio', hue='imgroup', ax=ax)
# ax.set(xlabel='Day of Year', ylabel='Green Ratio', title='Green Ratio Over Time')
# # %%
# fig, ax = plt.subplots(figsize=(12, 6))
# sns.lineplot(data=df, x='doys', y='green_ratio', hue='year', ax=ax)
# ax.set(xlabel='Day of Year', ylabel='Green Ratio', title='Green Ratio Over Time')
# %% for all pixels
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df, x='year', y='green_ratio', ax=ax)
ax.set(xlabel='Year', ylabel='Green Ratio', title='Green Ratio Distribution by Year')

# %%
# Create a figure with subplots for each imgroup
unique_imgroups = df['imgroup'].unique()
fig, axs = plt.subplots(len(unique_imgroups), 1, figsize=(14, 10), sharex=True)

# Plot data for each imgroup in separate subplots
handles, labels = None, None

for i, imgroup in enumerate(unique_imgroups):
    group_data = df[df['imgroup'] == imgroup]
    g = sns.lineplot(data=group_data, x='doys', y='green_ratio', hue='year', ax=axs[i])
    
    # Store handles and labels from the first plot to use for the shared legend
    if i == 0:
        handles, labels = axs[i].get_legend_handles_labels()
    
    # Remove individual legends
    axs[i].get_legend().remove()
    
    axs[i].set_ylabel('Green Ratio')
    axs[i].set_title(f'Green Ratio Over Time - {imgroup}')
    
    # Only set xlabel for the bottom subplot
    if i == len(unique_imgroups) - 1:
        axs[i].set_xlabel('Day of Year')
    else:
        axs[i].set_xlabel('')

# Add a single legend for all subplots
fig.legend(handles, labels, title="Year", loc='upper right', bbox_to_anchor=(1.15, 0.9))

plt.tight_layout()
# %% focus on west-facing images
west_df = df[df['imgroup'].str.contains('West')]

#%% Plot understory and birch ratios for west-facing images
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot understory ratio in top subplot
sns.lineplot(data=west_df, x='doys', y='understory_ratio', hue='year', ax=axs[0])
axs[0].set(ylabel='Understory Ratio', title='Understory Green Ratio - West-facing Images')
axs[0].legend(title='Year')

# Plot birch ratio in bottom subplot
sns.lineplot(data=west_df, x='doys', y='birch_ratio', hue='year', ax=axs[1])
axs[1].set(xlabel='Day of Year', ylabel='Birch Ratio', title='Birch Green Ratio - West-facing Images')
axs[1].legend(title='Year')

plt.tight_layout()

#%% plot understory_norm and birch_norm for west-facing images
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot understory norm in top subplot
sns.lineplot(data=west_df, x='doys', y='understory_norm', hue='year', ax=axs[0])
axs[0].set(ylabel='Understory Norm', title='Understory Green Norm - West-facing Images')
axs[0].legend(title='Year')
axs[0].set_xlim(125, 225)  # Set x-axis limits for the top subplot
axs[0].set_ylim(0.3, 0.5)  # Set y-axis limits for the top subplot

# Plot birch norm in bottom subplot
sns.lineplot(data=west_df, x='doys', y='birch_norm', hue='year', ax=axs[1])
axs[1].set(xlabel='Day of Year', ylabel='Birch Norm', title='Birch Green Norm - West-facing Images')
axs[1].legend(title='Year')
axs[1].set_xlim(125, 225)  # Set x-axis limits for the bottom subplot
axs[1].set_ylim(0.3, 0.5)  # Set y-axis limits for the bottom subplot

plt.tight_layout()
#%% focus on North-facing images
north_df = df[df['imgroup'].str.contains('North')]

#%% Plot understory and birch ratios for North-facing images
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot understory ratio in top subplot
sns.lineplot(data=north_df, x='doys', y='understory_ratio', hue='year', ax=axs[0])
axs[0].set(ylabel='Understory Ratio', title='Understory Green Ratio - North-facing Images')
axs[0].legend(title='Year')

# Plot birch ratio in bottom subplot
sns.lineplot(data=north_df, x='doys', y='birch_ratio', hue='year', ax=axs[1])
axs[1].set(xlabel='Day of Year', ylabel='Birch Ratio', title='Birch Green Ratio - North-facing Images')
axs[1].legend(title='Year')

plt.tight_layout()

#%% plot understory_norm and birch_norm for North-facing images
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot understory norm in top subplot
sns.lineplot(data=north_df, x='doys', y='understory_norm', hue='year', ax=axs[0])
axs[0].set(ylabel='Understory Norm', title='Understory Green Norm - North-facing Images')
axs[0].legend(title='Year')
axs[0].set_xlim(125, 225)  # Set x-axis limits for the top subplot
axs[0].set_ylim(0.3, 0.5)  # Set y-axis limits for the top subplot

# Plot birch norm in bottom subplot
sns.lineplot(data=north_df, x='doys', y='birch_norm', hue='year', ax=axs[1])
axs[1].set(xlabel='Day of Year', ylabel='Birch Norm', title='Birch Green Norm - North-facing Images')
axs[1].legend(title='Year')
axs[1].set_xlim(125, 225)  # Set x-axis limits for the bottom subplot
axs[1].set_ylim(0.3, 0.5)  # Set y-axis limits for the bottom subplot

plt.tight_layout()
# %%
