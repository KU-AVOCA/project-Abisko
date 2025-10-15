#%% 
import numpy as np
from scipy.io import savemat

#%%
img = np.load("/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/Tower thermal images/preview/all/npyimages/West-facing_2023-03-20_10.30.02.npy")
savemat("/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/Tower thermal images/preview/all/matimages/West-facing_2023-03-20_10.30.02.mat", {'img': img})
# %%
