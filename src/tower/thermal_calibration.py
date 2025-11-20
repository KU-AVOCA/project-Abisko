import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat
from scipy.io import savemat

vignetCorr = np.loadtxt("/home/geofsn/GitHub/project-Abisko/src/tower/vignetCorrLayerSN10600001_110x295_4dec.csv", delimiter=",")


folder = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/Tower thermal images/preview/all/matimages"   # raw string avoids issues with backslashes
folderOut = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/Tower thermal images/preview/all/vignetCorrectedImages"

# Loop through all files in the folder
for filename in sorted(os.listdir(folder)):
    if filename.endswith(".mat") and filename.startswith("W"):
        filepath = os.path.join(folder, filename)
        data = loadmat(filepath)   # load the .mat file into a dict
        tempData = data['thermal_image']
        correctedData = tempData-vignetCorr
        filepath = os.path.join(folderOut, filename)
        savemat(filepath, {"thermalImage": correctedData})
