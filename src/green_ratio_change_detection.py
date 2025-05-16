'''
This script processes time-lapse image data from a tower camera to analyze seasonal changes
in vegetation greenness (phenology) at the Abisko site. It performs the following steps:

1. Loads green ratio data extracted from RGB images, with support for multiple camera orientations.
2. Filters out images taken during night or low-light conditions using solar elevation calculations (via pvlib).
3. Focuses analysis on west-facing camera images, removing overexposed images after a specified date.
4. Aggregates green ratio data to daily means.
5. Smooths the daily green ratio time series using the Savitzky-Golay filter to reduce noise.
6. Detects key phenological dates for each year using derivative analysis on the smoothed signal:
    - Start of Season (SOS): Date of maximum positive rate of change (green-up).
    - Peak of Season (POS): Date of maximum greenness.
    - End of Season (EOS): Date of maximum negative rate of change (senescence).
7. Visualizes the results:
    - Time series plots of raw and smoothed green ratio data, with detected phenology dates annotated.
    - Day-of-year (DOY) plots for interannual comparison.
    - Optional: Derivative plots for a selected year to inspect detection logic.
8. Outputs summary tables and optionally saves results and figures.


Shunan Feng (shf@ign.ku.dk)
'''
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pvlib
import seaborn as sns
from scipy.signal import savgol_filter # Import Savitzky-Golay filter

sns.set_theme(style="darkgrid", font_scale=1.2)

#%%
# Define Abisko coordinates
SITE_LATITUDE = 68.34808742 # in decimal degrees
SITE_LONGITUDE = 19.05077561 # in decimal degrees
SITE_ELEVATION = 400  # meters above sea level (approximate)

# Function to determine daylight hours
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
        # print(f"Error calculating solar position for {timestamp}: {e}") # Optional: uncomment for debugging
        return False

# Load Tower Time Lapse Image Data
# !! Please update this path if it's different !!
csvfile = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/Tower RGB images/Data_greenessByShunan_kmeans_mean/results/green_ratio_kmeans.csv'
try:
    df_tower = pd.read_csv(csvfile)
except FileNotFoundError:
    print(f"Error: CSV file not found at {csvfile}")
    print("Please update the 'csvfile' variable with the correct path.")
    exit()


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
df_tower.dropna(subset=['datetime'], inplace=True) # Remove rows where datetime conversion failed
df_tower['year'] = df_tower['datetime'].dt.year
df_tower['month'] = df_tower['datetime'].dt.month
df_tower['doys'] = df_tower['datetime'].dt.dayofyear
df_tower['hour'] = df_tower['datetime'].dt.hour
df_tower['minute'] = df_tower['datetime'].dt.minute
df_tower['imgroup'] = df_tower['filename'].str.split('/').str[-3]

# Filter out night/dusk/dawn images
print("Total images before filtering:", len(df_tower))
df_tower['is_daytime'] = df_tower.apply(is_daytime, axis=1)
daytime_df_tower = df_tower[df_tower['is_daytime']].copy() # Use .copy() to avoid SettingWithCopyWarning

# Remove images taken after 2023-08-17 in west-facing camera due to overexposure
daytime_df_tower = daytime_df_tower[~((daytime_df_tower['datetime'] > pd.to_datetime("2023-08-17")) & (daytime_df_tower['imgroup'].str.contains('West')))]
print("Daytime images:", len(daytime_df_tower))
print(f"Removed {len(df_tower) - len(daytime_df_tower)} images taken during night or low-light conditions")

# Only keep west-facing images
west_df = daytime_df_tower[daytime_df_tower['imgroup'].str.contains('West')].copy() # Use .copy()
west_df['date'] = west_df['datetime'].dt.date
west_df['date'] = pd.to_datetime(west_df['date']) # Convert date back to datetime for proper sorting/indexing

# Calculate daily mean green ratio
daily_green_ratio = west_df.groupby('date')['green_ratio'].mean().reset_index()
daily_green_ratio.sort_values('date', inplace=True)
daily_green_ratio['year'] = daily_green_ratio['date'].dt.year

# --- Data Smoothing using Savitzky-Golay Filter ---
# Fill potential NaNs before applying the filter
daily_green_ratio['green_ratio_filled'] = daily_green_ratio['green_ratio'].ffill().bfill()

# Apply Savitzky-Golay filter
# Parameters:
# window_length: The length of the filter window (must be a positive odd integer).
# polyorder: The order of the polynomial used to fit the samples (must be less than window_length).
# Choose parameters based on desired smoothing level. Larger window_length -> more smoothing.
# Common values: window_length=7, 11, 15; polyorder=2, 3
window_length = 11 # Example: 11 days window
polyorder = 3    # Example: polynomial order 3

# Check if the dataset is long enough for the chosen window length
if len(daily_green_ratio) >= window_length:
    daily_green_ratio['green_ratio_smooth'] = savgol_filter(
        daily_green_ratio['green_ratio_filled'],
        window_length=window_length,
        polyorder=polyorder,
        mode='interp' # Interpolate at boundaries
    )
    print(f"\nApplied Savitzky-Golay filter (window={window_length}, order={polyorder}).")
else:
    # Handle cases where data is too short for the filter
    print(f"\nWarning: Data length ({len(daily_green_ratio)}) is less than Savitzky-Golay window length ({window_length}). Using original (filled) data instead of smoothing.")
    daily_green_ratio['green_ratio_smooth'] = daily_green_ratio['green_ratio_filled']


# Save the data including the smoothed column
# daily_green_ratio.to_csv('daily_green_ratio_smoothed_savgol.csv', index=False)

# --- Phenology Detection using Derivative Analysis ---
# %%
phenology_dates = {} # Dictionary to store dates for each year: {'year': {'start': date, 'peak': date, 'end': date}}
unique_years = sorted(daily_green_ratio['year'].unique())

print(f"\n--- Phenology Detection Results (Derivative Analysis on SavGol Smoothed Data) ---")

for year in unique_years:
    print(f"\nProcessing Year: {year}")
    # Filter data for the current year
    yearly_data = daily_green_ratio[daily_green_ratio['year'] == year].copy().reset_index(drop=True)

    # Ensure enough data points for derivative calculation
    min_points_for_deriv = 5 # Need some points to get meaningful derivatives
    if len(yearly_data) < min_points_for_deriv:
        print(f"Skipping year {year} due to insufficient data ({len(yearly_data)} points) for derivative analysis.")
        phenology_dates[year] = {'start': pd.NaT, 'peak': pd.NaT, 'end': pd.NaT}
        continue

    # Use the smoothed green ratio
    smoothed_signal = yearly_data['green_ratio_smooth'].values
    dates = yearly_data['date']
    doys = yearly_data['date'].dt.dayofyear # Use DOY for easier interpretation

    # Check for constant signal
    if np.std(smoothed_signal) < 1e-9:
        print(f"Skipping year {year} because the smoothed signal is constant.")
        phenology_dates[year] = {'start': pd.NaT, 'peak': pd.NaT, 'end': pd.NaT}
        continue

    # Calculate first derivative (rate of change)
    first_derivative = np.gradient(smoothed_signal)

    # Calculate second derivative (rate of change of the rate of change)
    second_derivative = np.gradient(first_derivative)

    # --- Identify Phenological Dates ---
    try:
        # Start of Season (SOS): Maximum of the first derivative (fastest green-up)
        # Alternative: Maximum of the second derivative (maximum acceleration) - choose one or combine logic
        sos_index = np.argmax(first_derivative)
        sos_date = dates.iloc[sos_index]
        print(f"  SOS (Max 1st Deriv): DOY {doys.iloc[sos_index]}, Date {sos_date.strftime('%Y-%m-%d')}")

        # End of Season (EOS): Minimum of the first derivative (fastest green-down/senescence)
        # Alternative: Minimum of the second derivative (maximum deceleration)
        eos_index = np.argmin(first_derivative)
        eos_date = dates.iloc[eos_index]
        print(f"  EOS (Min 1st Deriv): DOY {doys.iloc[eos_index]}, Date {eos_date.strftime('%Y-%m-%d')}")

        # Peak of Season (POS): Where the smoothed signal is maximum
        # Alternative: Where the first derivative is closest to zero between SOS and EOS
        pos_index = np.argmax(smoothed_signal)
        pos_date = dates.iloc[pos_index]
        print(f"  POS (Max Signal)   : DOY {doys.iloc[pos_index]}, Date {pos_date.strftime('%Y-%m-%d')}")

        # Basic validation: Check if dates are in logical order
        if not (sos_date < pos_date < eos_date):
             print(f"  Warning: Phenological dates for year {year} are not in expected order (SOS < POS < EOS). Review derivatives.")
             # You might add logic here to refine based on second derivative or other constraints

        phenology_dates[year] = {'start': sos_date, 'peak': pos_date, 'end': eos_date}

    except Exception as e:
        print(f"An error occurred during derivative analysis for year {year}: {e}")
        phenology_dates[year] = {'start': pd.NaT, 'peak': pd.NaT, 'end': pd.NaT}


# --- Convert results to DataFrame for easier plotting/saving ---
phenology_df_list = []
for year, dates_dict in phenology_dates.items():
    if pd.notna(dates_dict['start']): # Only add if valid dates were found
        phenology_df_list.append({
            'year': year,
            'sos_date': dates_dict['start'],
            'pos_date': dates_dict['peak'],
            'eos_date': dates_dict['end'],
            'sos_doy': dates_dict['start'].dayofyear,
            'pos_doy': dates_dict['peak'].dayofyear,
            'eos_doy': dates_dict['end'].dayofyear
        })
phenology_df = pd.DataFrame(phenology_df_list)
print("\n--- Summary of Detected Phenology Dates ---")
print(phenology_df)
# Optional: Save phenology dates
# phenology_df.to_csv('phenology_dates_derivative.csv', index=False)


# --- Visualization (Combined Plot - Adapted for Phenology Dates) ---
fig, ax = plt.subplots(figsize=(18, 7))

# Plot the ORIGINAL full time series (lighter)
ax.plot(daily_green_ratio['date'], daily_green_ratio['green_ratio'], label='Daily Mean Green Ratio (Raw)', alpha=0.4, color='grey', zorder=1)

# Plot the SMOOTHED full time series (darker)
ax.plot(daily_green_ratio['date'], daily_green_ratio['green_ratio_smooth'], label=f'Smoothed Green Ratio (SavGol w={window_length}, p={polyorder})', alpha=0.9, color='black', zorder=2)

# Scatter plot the points colored by year for clarity (using smoothed data for position)
cmap = plt.get_cmap('viridis', len(unique_years))
for i, year in enumerate(unique_years):
    yearly_data = daily_green_ratio[daily_green_ratio['year'] == year]
    ax.scatter(yearly_data['date'], yearly_data['green_ratio_smooth'], color=cmap(i), label=f'Year {year} (Smoothed)', s=10, zorder=3)


# Plot detected phenology points from all years
if not phenology_df.empty:
    # Get corresponding smoothed green ratio values for annotation height
    plot_dates = pd.concat([phenology_df['sos_date'], phenology_df['pos_date'], phenology_df['eos_date']]).dropna().unique()
    cp_data = daily_green_ratio[daily_green_ratio['date'].isin(plot_dates)]

    # Plot SOS points
    sos_points = phenology_df.dropna(subset=['sos_date'])
    if not sos_points.empty:
        sos_plot_data = cp_data[cp_data['date'].isin(sos_points['sos_date'])]
        ax.scatter(sos_plot_data['date'], sos_plot_data['green_ratio_smooth'],
                   color='lime', marker='^', s=120, label='Start of Season (SOS)', zorder=5, edgecolors='black')

    # Plot POS points
    pos_points = phenology_df.dropna(subset=['pos_date'])
    if not pos_points.empty:
        pos_plot_data = cp_data[cp_data['date'].isin(pos_points['pos_date'])]
        ax.scatter(pos_plot_data['date'], pos_plot_data['green_ratio_smooth'],
                   color='gold', marker='s', s=120, label='Peak of Season (POS)', zorder=5, edgecolors='black')

    # Plot EOS points
    eos_points = phenology_df.dropna(subset=['eos_date'])
    if not eos_points.empty:
        eos_plot_data = cp_data[cp_data['date'].isin(eos_points['eos_date'])]
        ax.scatter(eos_plot_data['date'], eos_plot_data['green_ratio_smooth'],
                   color='brown', marker='v', s=120, label='End of Season (EOS)', zorder=5, edgecolors='black')

else:
    print("\nNo phenology dates detected to plot.")


ax.set_title('Phenology Detection using Derivative Analysis (West Camera - SavGol Smoothed Data)')
ax.set_xlabel('Date')
ax.set_ylabel('Mean Green Ratio')

# Adjust legend position
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))


plt.xticks(rotation=45)
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
plt.show()

# Optional: Save the plot
# fig.savefig('../print/green_ratio_phenology_derivative_yearly.png', dpi=300, bbox_inches='tight')
# fig.savefig('../print/green_ratio_phenology_derivative_yearly.pdf', dpi=300, bbox_inches='tight')


# --- Visualization (DOY Plot - Adapted for Phenology Dates) ---
fig, ax = plt.subplots(figsize=(12, 7))

# Add DOY column if it doesn't exist
if 'doys' not in daily_green_ratio.columns:
     daily_green_ratio['doys'] = daily_green_ratio['date'].dt.dayofyear

# Plot the daily green ratio vs DOY, colored by year (SMOOTHED)
cmap = plt.get_cmap('viridis', len(unique_years))
for i, year in enumerate(unique_years):
    yearly_data = daily_green_ratio[daily_green_ratio['year'] == year]
    # Plot original data lightly
    ax.plot(yearly_data['doys'], yearly_data['green_ratio'], linestyle='-', color=cmap(i), alpha=0.3, markersize=4)
    # Plot smoothed data more prominently
    ax.plot(yearly_data['doys'], yearly_data['green_ratio_smooth'], marker='.', linestyle='-', label=f'Year {year}', color=cmap(i), alpha=0.9, markersize=4)


# Mark the detected phenology points on the plot
if not phenology_df.empty:
     # Get smoothed values at the phenology dates for plotting
     pheno_plot_data = daily_green_ratio[daily_green_ratio['date'].isin(plot_dates)].copy()
     if 'doys' not in pheno_plot_data.columns:
         pheno_plot_data['doys'] = pheno_plot_data['date'].dt.dayofyear

     # Merge with phenology_df to get DOYs and dates aligned
     sos_points_doy = phenology_df.dropna(subset=['sos_date'])[['sos_date', 'sos_doy']].rename(columns={'sos_date':'date', 'sos_doy':'doys'})
     pos_points_doy = phenology_df.dropna(subset=['pos_date'])[['pos_date', 'pos_doy']].rename(columns={'pos_date':'date', 'pos_doy':'doys'})
     eos_points_doy = phenology_df.dropna(subset=['eos_date'])[['eos_date', 'eos_doy']].rename(columns={'eos_date':'date', 'eos_doy':'doys'})

     sos_plot = pd.merge(sos_points_doy, pheno_plot_data[['date', 'green_ratio_smooth']], on='date', how='left')
     pos_plot = pd.merge(pos_points_doy, pheno_plot_data[['date', 'green_ratio_smooth']], on='date', how='left')
     eos_plot = pd.merge(eos_points_doy, pheno_plot_data[['date', 'green_ratio_smooth']], on='date', how='left')

     # Plot SOS
     if not sos_plot.empty:
         ax.scatter(sos_plot['doys'], sos_plot['green_ratio_smooth'],
                    color='lime', marker='^', s=120, label='Start of Season (SOS)', zorder=5, edgecolors='black')
     # Plot POS
     if not pos_plot.empty:
         ax.scatter(pos_plot['doys'], pos_plot['green_ratio_smooth'],
                    color='gold', marker='s', s=120, label='Peak of Season (POS)', zorder=5, edgecolors='black')
     # Plot EOS
     if not eos_plot.empty:
         ax.scatter(eos_plot['doys'], eos_plot['green_ratio_smooth'],
                    color='brown', marker='v', s=120, label='End of Season (EOS)', zorder=5, edgecolors='black')


ax.set_title('Daily Mean Green Ratio vs. Day of Year (West Camera - SavGol Smoothed)')
ax.set_xlabel('Day of Year (DOY)')
ax.set_ylabel('Mean Green Ratio')
ax.set_xlim(0, 366) # Set x-axis limits for DOY

# Adjust legend position
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))


plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
plt.show()

# Optional: Save the plot
# fig.savefig('../print/green_ratio_phenology_derivative_doy.png', dpi=300, bbox_inches='tight')
# fig.savefig('../print/green_ratio_phenology_derivative_doy.pdf', dpi=300, bbox_inches='tight')

# --- Optional: Plot Derivatives for a Specific Year ---
year_to_plot = 2022 # Choose a year
if year_to_plot in unique_years:
    yearly_data = daily_green_ratio[daily_green_ratio['year'] == year_to_plot].copy().reset_index(drop=True)
    if len(yearly_data) >= min_points_for_deriv:
        smoothed_signal = yearly_data['green_ratio_smooth'].values
        first_derivative = np.gradient(smoothed_signal)
        second_derivative = np.gradient(first_derivative)
        doys = yearly_data['date'].dt.dayofyear

        fig_deriv, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        axes[0].plot(doys, smoothed_signal, label='Smoothed Green Ratio', color='black')
        axes[0].set_ylabel('Green Ratio')
        axes[0].set_title(f'Derivatives for Year {year_to_plot}')
        axes[0].grid(True)

        axes[1].plot(doys, first_derivative, label='1st Derivative', color='blue')
        axes[1].axhline(0, color='grey', linestyle='--', lw=1)
        axes[1].set_ylabel('1st Derivative')
        axes[1].grid(True)

        axes[2].plot(doys, second_derivative, label='2nd Derivative', color='red')
        axes[2].axhline(0, color='grey', linestyle='--', lw=1)
        axes[2].set_ylabel('2nd Derivative')
        axes[2].set_xlabel('Day of Year (DOY)')
        axes[2].grid(True)

        # Mark detected points on derivative plots if available
        if year_to_plot in phenology_dates and pd.notna(phenology_dates[year_to_plot]['start']):
            sos_doy = phenology_dates[year_to_plot]['start'].dayofyear
            pos_doy = phenology_dates[year_to_plot]['peak'].dayofyear
            eos_doy = phenology_dates[year_to_plot]['end'].dayofyear
            axes[0].axvline(sos_doy, color='lime', linestyle=':', label='SOS')
            axes[0].axvline(pos_doy, color='gold', linestyle=':', label='POS')
            axes[0].axvline(eos_doy, color='brown', linestyle=':', label='EOS')
            axes[1].axvline(sos_doy, color='lime', linestyle=':')
            axes[1].axvline(pos_doy, color='gold', linestyle=':')
            axes[1].axvline(eos_doy, color='brown', linestyle=':')
            axes[2].axvline(sos_doy, color='lime', linestyle=':')
            axes[2].axvline(pos_doy, color='gold', linestyle=':')
            axes[2].axvline(eos_doy, color='brown', linestyle=':')
            axes[0].legend()


        plt.tight_layout()
        plt.show()


# %%
