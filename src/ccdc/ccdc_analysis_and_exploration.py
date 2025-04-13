#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", font_scale=1.5)
# %% load data
df_origional = pd.read_csv("/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/CCDC/data/E1_GCC_origional.csv")
df_predicted = pd.read_csv("/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/CCDC/data/E1_GCC_predicted.csv")
df_origional[df_origional.GCC >1] = np.nan
df_origional[df_origional.GCC <0] = np.nan

# %% data preprocessing 
df_origional = df_origional.rename(columns={"system:time_start": "date"})
df_predicted = df_predicted.rename(columns={"system:time_start": "date"})
df_origional["date"] = pd.to_datetime(df_origional["date"], format="%d-%b-%y")
df_predicted["date"] = pd.to_datetime(df_predicted["date"], format="%d-%b-%y")
df_origional = df_origional.groupby(pd.Grouper(key="date", freq="d"))["GCC"].mean().reset_index()
df = pd.merge(
    df_origional,
    df_predicted,
    on=["date"]
    # suffixes=("_origional", "_predicted"),
)
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["day"] = df["date"].dt.day

# %% create time series plot to show the CCDC model fitted and predicted values with observations
fig, ax = plt.subplots(figsize=(20, 6))  # Increased width from 20 to 24
# Create line plots
line1 = sns.lineplot(
    data=df[df["year"] < 2022],
    x="date",
    y="GCC_predicted",
    label="GCC fitted",
    ax=ax
)
line2 = sns.lineplot(
    data=df[df["year"] >= 2022],
    x="date",
    y="GCC_predicted",
    label="GCC predicted",
    dashes=(2, 2),
    ax=ax
)
# Create scatter plot with its own legend
scatter = sns.scatterplot(
    data=df,
    x="date",
    y="GCC",
    hue="month",
    palette="Paired",
    ax=ax
)
ax.set(
    xlim=pd.to_datetime(["2014-01-01", "2024-12-31"]),
    ylim=(0.25, 0.5),
    xlabel="Date",
    ylabel="Green Chromatic Coordinate (GCC)"
    )
# add vertical line to show the start of the predicted data
ax.axvline(pd.to_datetime("2022-01-01"), color="black", linestyle="-.", linewidth=1)
ax.text(pd.to_datetime("2021-12-01"), 0.475, "<--- base period", ha='right', color=sns.color_palette()[0])
ax.text(pd.to_datetime("2022-02-01"), 0.475, "predicted period --->", ha='left', color=sns.color_palette()[1])

# Remove default legends
ax.get_legend().remove()

# Create custom legends
# Line plot legend
line_handles = [line1.get_lines()[0], line2.get_lines()[1]]
line_labels = ["fitted", "predicted"]
first_legend = ax.legend(line_handles, line_labels, ncol=2,
                          loc='upper left', frameon=True)

# Add scatter plot legend with title
ax.add_artist(first_legend)
scatter_handles, scatter_labels = scatter.get_legend_handles_labels()
scatter_labels = scatter_labels[2:]  # Skip the line labels
scatter_handles = scatter_handles[2:]  # Skip the line handles
ax.legend(scatter_handles, scatter_labels, title="Month", ncol=12,
          loc='upper left',  bbox_to_anchor=(0.005, 1.2), frameon=True)
# fig.savefig("../../print/CCDCatE1.png", dpi=300, bbox_inches="tight")
# fig.savefig("../../print/CCDCatE1.pdf", dpi=300, bbox_inches="tight")
# %% calculate the relative deviation of the predicted values from the observed values
df["diff"] = df["GCC"] - df["GCC_predicted"] 
df["deviation_norm"] = df["diff"] / df["GCC_predicted"]

fig, ax = plt.subplots(figsize=(20, 6))  
sns.scatterplot(
    data=df.dropna(),
    x="date",
    y="deviation_norm",
    ax=ax
)
plt.ylim(-0.4, 0.4)
#%%
df_summer = df[df["month"].isin([6, 7, 8])]
dfstd = df_summer["deviation_norm"].std()
df_annual = df_summer.groupby("year")["deviation_norm"].mean().reset_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Bar plot for annual average deviation norm
sns.barplot(
    data=df_annual,
    x="year",
    y="deviation_norm",
    ax=ax2,
    label="Normal Condition",
)
sns.barplot(
    data=df_annual[df_annual["deviation_norm"].abs() > dfstd],
    x="year",
    y="deviation_norm",
    ax=ax2,
    label="Anomalous Condition",
)

ax2.legend(loc="lower left")
ax2.set(
    xlabel="Year",
    ylabel="Annual Average Deviation Norm",
    title="Annual Average Deviation Norm",
)

# Scatter plot for summer deviation norm
sns.scatterplot(
    data=df_summer,
    x="date",
    y="deviation_norm",
    ax=ax1,
)
ax1.set(
    xlabel="Date",
    ylabel="Deviation Norm",
    title="Summer Deviation Norm",
)
plt.tight_layout()

ax2.legend(loc="lower left")
# ax.set(
#     xlabel="Year",
#     ylabel="Annual Average Deviation Norm",
#     title="Annual Average Deviation Norm"
# )
# plt.show()
# %%
df["obs-obsmean"] = df["GCC"] - df["GCC"].mean()
df["test_metrics"] = df["diff"]/ df["obs-obsmean"]
fig, ax = plt.subplots(figsize=(20, 6))
sns.scatterplot(
    data=df.dropna(),
    x="date",
    y="test_metrics",
    ax=ax
)
plt.ylim(-10, 10)
# %%
