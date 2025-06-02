#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cmocean 

sns.set_theme(style="darkgrid", font_scale=1.5)
#%%
df = pd.read_csv("book.csv", delimiter=";")
# %%
sns.barplot(
    data=df,
    x="Class name",
    y="Images",
    hue="Precision",
    palette=cmocean.cm.thermal
)
# remove legend
plt.legend([],[], frameon=False)
# show colorbar
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmocean.cm.thermal), ax=plt.gca())
cbar.set_label("Precision")
# rotate x-axis labels
plt.xticks(rotation=90)
plt.xlabel("Class name")
plt.ylabel("Number of images")
# %%
