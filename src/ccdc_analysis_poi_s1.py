#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

sns.set_theme(style="darkgrid", font_scale=1.5)
# %% load data
df_origional = pd.read_csv("/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/CCDC/data/E1_S1ratio_desc_origional.csv")
df_predicted = pd.read_csv("/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/CCDC/data/E1_S1ratio_desc_predicted.csv")


# %% data preprocessing 
df_origional = df_origional.rename(columns={"system:time_start": "date"})
df_predicted = df_predicted.rename(columns={"system:time_start": "date"})
df_origional["date"] = pd.to_datetime(df_origional["date"], format="%d-%b-%y")
df_predicted["date"] = pd.to_datetime(df_predicted["date"], format="%d-%b-%y")
df_origional = df_origional.groupby(pd.Grouper(key="date", freq="d"))["S1ratio"].mean().reset_index()
df = pd.merge(
    df_origional,
    df_predicted,
    on=["date"]
    # suffixes=("_origional", "_predicted"),
).dropna()
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["day"] = df["date"].dt.day
# quick statistics S1ratio vs S1ratio_predicted
slope, intercept, r_value, p_value, std_err = stats.linregress(df[df.year<2022].S1ratio, df[df.year<2022].S1ratio_predicted)
sns.regplot(data=df[df.year<2022], x="S1ratio", y="S1ratio_predicted")
print(f"R: {r_value:.4f}")
print(f"p-value: {p_value:.4f}")

df["diff"] = df["S1ratio"] - df["S1ratio_predicted"] 
df["deviation_norm"] = df["diff"] / df["S1ratio_predicted"]
df_summer = df[df["month"].isin([6, 7, 8])]
# dfstd = df_summer["deviation_norm"].std()
df_annual = df_summer.groupby("year")["deviation_norm"].mean().reset_index()
# %% create time series plot to show the CCDC model fitted and predicted values with observations
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15))

# Top plot - CCDC model fitted and predicted values
line1 = sns.lineplot(
    data=df[df["year"] < 2022],
    x="date",
    y="S1ratio_predicted",
    label="S1ratio fitted",
    ax=ax1
)
line2 = sns.lineplot(
    data=df[df["year"] >= 2022],
    x="date",
    y="S1ratio_predicted",
    label="S1ratio predicted",
    dashes=(2, 2),
    ax=ax1
)
scatter = sns.scatterplot(
    data=df,
    x="date",
    y="S1ratio",
    hue="month",
    palette="Paired",
    ax=ax1
)
ax1.set(
    xlim=pd.to_datetime(["2015-01-01", "2024-12-31"]),
    # ylim=(0.25, 0.5),
    ylabel="VH / VV",
    xlabel="Date",
)
ax1.axvline(pd.to_datetime("2022-01-01"), color="black", linestyle="-.", linewidth=1)
ax1.text(pd.to_datetime("2021-12-01"), 0.475, "<--- base period", ha='right', color=sns.color_palette()[0])
ax1.text(pd.to_datetime("2022-02-01"), 0.475, "predicted period --->", ha='left', color=sns.color_palette()[1])

# Remove default legends
ax1.get_legend().remove()

# Custom legends for top plot
line_handles = [line1.get_lines()[0], line2.get_lines()[1]]
line_labels = ["fitted", "predicted"]
first_legend = ax1.legend(line_handles, line_labels, ncol=2,
                          loc='upper left', frameon=True)

# Add scatter plot legend with title
ax1.add_artist(first_legend)
scatter_handles, scatter_labels = scatter.get_legend_handles_labels()
scatter_labels = scatter_labels[2:]  # Skip the line labels
scatter_handles = scatter_handles[2:]  # Skip the line handles
ax1.legend(scatter_handles, scatter_labels, title="Month", ncol=12,
          loc='upper left', bbox_to_anchor=(0.005, 1.27), frameon=True)

# Middle plot - Summer deviation norm
sns.scatterplot(
    data=df[df.deviation_norm>=0],
    x="date",
    y="deviation_norm",
    ax=ax2,
    color=sns.palettes.color_palette()[0]
)
sns.scatterplot(
    data=df[df.deviation_norm<0],
    x="date",
    y="deviation_norm",
    ax=ax2,
    color=sns.palettes.color_palette()[1]
)
ax2.set(
    xlabel="Date",
    ylabel="Relative Deviation (RD)",
    # title="Summer Deviation Norm",
)
ax2.set(
    xlim=pd.to_datetime(["2015-01-01", "2024-12-31"]),
    ylim=(-0.3, 0.3)
)

# Bottom plot - Annual average deviation norm
sns.barplot(
    data=df_annual,
    x="year",
    y="deviation_norm",
    color="#297383",
    ax=ax3,
    label="Normal Condition",
)
sns.barplot(
    data=df_annual[df_annual["deviation_norm"].abs() > df_annual["deviation_norm"].std()],
    x="year",
    y="deviation_norm",
    color="#832900",
    ax=ax3,
    label="Anomalous Condition",
)
ax3.axhline(df_annual["deviation_norm"].std(), color="black", linestyle="--", linewidth=1, label=r"+1$\sigma$")
ax3.axhline(-df_annual["deviation_norm"].std(), color="black", linestyle="--", linewidth=1, label=r"$-$1$\sigma$")
ax3.legend(loc="lower center", ncol=2)
ax3.set(
    xlabel="Year",
    ylabel="Annual Average Summer RD",
    # title="Annual Average Deviation Norm"
    # xlim=(2014, 2024),
)
# Add subplot annotations
ax1.annotate('a)', xy=(0.03, 0.1), xycoords='axes fraction')
ax2.annotate('b)', xy=(0.03, 0.1), xycoords='axes fraction')
ax3.annotate('c)', xy=(0.03, 0.1), xycoords='axes fraction')
# plt.tight_layout()
# fig.savefig("../print/CCDCatE1_combined.png", dpi=300, bbox_inches="tight")
# fig.savefig("../print/CCDCatE1_combined.pdf", dpi=300, bbox_inches="tight")




# %%
