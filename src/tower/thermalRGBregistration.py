#%% 
import numpy as np
from scipy.io import savemat

#%%
img = np.load("/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/Tower thermal images/preview/all/npyimages/North-facing_2022-03-20_12.30.01.npy")
savemat("/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/Tower thermal images/preview/all/matimages/North-facing_2022-03-20_12.30.01.mat", {'image_thermal': img})
# %%
