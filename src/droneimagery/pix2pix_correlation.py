#%%
import os
import vaex as vx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmocean 
import glob

sns.set_theme(style="darkgrid", font_scale=1.5)
# %%
# h5folder = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/drone/pix2pix/"
h5folder = "/home/geofsn/data/abiskoproject/"
h5files = glob.glob(os.path.join(h5folder, "*.h5"), recursive=True)
# load list of h5 files as vaex dataframe
# df = vx.open(h5files[0])
# # load all h5 files into a single vaex dataframe
# for h5file in h5files[1:]:
#     df = vx.concat([df, vx.open(h5file)])
df = vx.open_many(h5files)    
# %%
fig, ax = plt.subplots(figsize=(7, 8))
# plt.plot([0, -1], [1, 1], 'w--', lw=2, label='1:1 Line')
ax.set(xlim=(0,1), ylim=(-1,1))
ax.annotate('n: {}'.format(df.GCC.count()), xy=(0.5, 0.1), xycoords='axes fraction')
df.viz.heatmap(df.GCC, df.NDVI, what=np.log(vx.stat.count()), show=True, colormap=cmocean.cm.haline,
               xlabel='GCC', ylabel='NDVI')
ax.set_aspect('equal')
fig.savefig("print/GCC_NDVI_correlation_all1.png", dpi=300)
# %%
# log transform GCC
# df['logGCC'] = np.log(df.GCC)
# fig, ax = plt.subplots(figsize=(7, 8))
# # plt.plot([0, -1], [1, 1], 'w--', lw=2, label='1:1 Line')
# # ax.set(xlim=(0,1), ylim=(-1,1))
# df.viz.heatmap(df.logGCC, df.NDVI, what=np.log(vx.stat.count()), show=True, colormap=cmocean.cm.haline,
#                xlabel='GCC (log)', ylabel='NDVI')
# ax.set_aspect('equal')
# fig.savefig("print/logGCC_NDVI_correlation_all.png", dpi=300)
# # %%
# # filter out data with GCC < 0.38
# df_filtered = df[df.GCC > 0.38]
# fig, ax = plt.subplots(figsize=(7, 8))
# # plt.plot([0, -1], [1, 1], 'w--', lw=2, label='1:1 Line')
# # ax.set(xlim=(0,1), ylim=(-1,1))
# df_filtered.viz.heatmap(df_filtered.logGCC, df_filtered.NDVI, what=np.log(vx.stat.count()), show=True, colormap=cmocean.cm.haline,
#                xlabel='logGCC', ylabel='NDVI')
# ax.set_aspect('equal')
# fig.savefig("print/logGCC_NDVI_correlation_filtered.png", dpi=300)
# %%
