# Multi-source Remote Sensing of Insect Defoliation Events in Abisko from Point to Regional Scales
This repository contains the code and data for the paper "Multi-source Remote Sensing of Insect Defoliation Events in Abisko from Point to Regional Scales".
A web application [defoliation detector](https://ku-avoca.projects.earthengine.app/view/defoliationdetector) is available at [https://ku-avoca.projects.earthengine.app/view/defoliationdetector](https://ku-avoca.projects.earthengine.app/view/defoliationdetector) for inspecting defoliation events in Abisko region. We hope to expand the web application to larger areas in the future.


The code is organized into different sections for data processing, analysis, and visualization.

## Data Preparation
The data preparation section includes scripts for downloading and processing the data used in the analysis. There are four main sources of data:
1. **Close-up**: Close-up images of the sampling plots.
2. **Time-lapse**: Time-lapse images inside the sampling area.
3. **Drone**: Drone images of the sampling area.
4. **Satellite**: Harmonized Landsat and Sentinel-2 (HLS) data covering the Abisko region.

Data and processed results are available in KU ERDA repository: [AVOCA/Supplementary4manuscript_2025_Abisko_Defoliation](https://erda.ku.dk/archives/ea08b4082c8c4c817bee04aabce831a3/published-archive.html).
HLS data are available on Google Earth Engine (GEE). 

The code for downloading and processing the data is organized into different steps:
- [src/closeup/green_ratio_detector_GCCdynamicthreshold.py](src/closeup/green_ratio_detector_GCCdynamicthreshold.py): This step calculates the greenness ratio threshold for the close-up images. It converts the images to GCC values and applies Otsu's method to find the optimal threshold for each image. The optimal threshold is the mean +- 1 standard deviation of the obtained thresholds.
    - Processed images and the summary table (csv file) are saved in ERDA repository: 
        - `AVOCA/Supplementary4manuscript_2025_Abisko_Defoliation/CloseupImages/closeup_green_ratio_mean`
        - `AVOCA/Supplementary4manuscript_2025_Abisko_Defoliation/CloseupImages/closeup_green_ratio_stdplus`
        - `AVOCA/Supplementary4manuscript_2025_Abisko_Defoliation/CloseupImages/closeup_green_ratio_stdminus`
- [src/closeup/green_ratio_detector_GCC_towerRGB_simple.py](src/src/closeup/green_ratio_detector_GCC_towerRGB_simple.py): This step applies the greenness ratio threshold to the time-lapse images. It converts the images to GCC values and applies the optimal threshold obtained in the previous step to classify the images into healthy (green) and unhealthy/non-vegetation categories. It also filters out images acquired during poor lighting conditions (solar angle altitude < 5 degrees).
    - Processed images and the summary table (csv file) are saved in ERDA repository: 
        - `AVOCA/Supplementary4manuscript_2025_Abisko_Defoliation/TowerRGBimages/Data_greenessByShunan_simple_mean`
        - `AVOCA/Supplementary4manuscript_2025_Abisko_Defoliation/TowerRGBimages/Data_greenessByShunan_simple_stdplus`
        - `AVOCA/Supplementary4manuscript_2025_Abisko_Defoliation/TowerRGBimages/Data_greenessByShunan_simple_stdminus`
- [src/ccdc/ccdcRun.js](src/ccdc/ccdcRun.js): This step runs the Continuous Change Detection and Classification (CCDC) algorithm on the HLS data. Simply copy the code and paste it into the GEE code editor. The CCDC image will be saved in your GEE asset folder.
- [src/ccdc/ccdcvis_poi.js](src/ccdc/ccdcvis_poi.js): This step extract the time sereis of observed and synthetic GCC values from a point of interest (POI) in the CCDC image. The POI is defined by the user. The time series plots will be displayed in the GEE code editor, which can be downloaded as csv files. The paper uses E1 sample site as the POI. 
    - The time series plots are saved in the ERDA repository: 
        - `AVOCA/Supplementary4manuscript_2025_Abisko_Defoliation/CCDC/data/E1_GCC_predictedCombinePre.csv` (synthetic GCC)
        - `AVOCA/Supplementary4manuscript_2025_Abisko_Defoliation/CCDC/data/E1_GCC_origional.csv` (observed GCC)
- [src/ccdc/ccdcvis_raster.js](src/ccdc/ccdcvis_raster.js): This step creates observed and synthetic GCC images in the regions of interest (ROI). Those daily images will be exported to your Googel Drive folders. The processed images are available in the ERDA repository (`AVOCA/Supplementary4manuscript_2025_Abisko_Defoliation/CCDC/data/GCCdaily`). 
The band designations are as follows (more details in the paper):
    - `band1: GCC_predicted` (synthetic GCC)
    - `band2: GCC_RMSE` (RMSE from the CCDC algorithm, but not used in the paper)
    - `band3: GCC` (observed GCC)
    - `band4: conditionScore` (condition score, but not used in the paper)
    - `band5: relativeDeviation` (relative deviation)
- [src/ccdc/ccdc_defoliation_analysis.py](src/ccdc/ccdc_defoliation_analysis.py): This step analyzes the images exported from GEE. It calculates the annual summer defoliation intensity as described in the paper. 
    - The results are saved as numpy arrays in the ERDA repository: 
        - `AVOCA/Supplementary4manuscript_2025_Abisko_Defoliation/CCDC/data/cs_annual.npy`

## Data Analysis 
The data analysis section includes scripts for analyzing the data and generating figures for the paper. The analysis is organized into different steps:
- [src/ccdc_analysis_poi.py](src/ccdc_analysis_poi.py): This step analyzes the time series of observed and synthetic GCC values from the POI. It calculates the relative deviations and the annual summer defoliation intensity. This script corresponds to Fig.2 in the paper. 
- [src/insect_analysis.py](src/insect_analysis.py): This step focuses on the insect abundance data and the interactions with other parameters including HLS GCC derived defoliation intensity, time-lapse and close up images derived Greenness Ration, and total BVOC emmissions. This script corresponds to Fig.3 in the paper.
- [src/defoliation_analysis.py](src/defoliation_analysis.py): This step focuses on the defoliation mapping. It plots the annual summer defoliation intensity from 2013 to 2024. This script corresponds to Fig.5 in the paper.
- Other figures (Fig.1, Fig.4, and Fig.6) were produced using QGIS or inkscape, and are not included in this repository. There are also some exploratory scripts that are not included in the paper, but are available in the `src` folder.

## References
The manuscript is current submitted for peer review. The reference will be updated once the paper is accepted.
If you use this code or data in your research, please cite the following paper and data repository:
