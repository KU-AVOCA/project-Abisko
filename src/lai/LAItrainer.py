#%%

import argparse
import os
import sys
import warnings
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
import rasterio
from rasterio.sample import sample_gen
from PIL import Image

#%%
dflai = pd.read_excel("/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/LAI/leafAreaIndex_compiledAbisko2021.xlsx", sheet_name='Sheet1post')
# rename corrected licor to lai
dflai = dflai.rename(columns={'corrected licor':'lai'})
# convert coordinates from DMS to decimal degrees
dflai['lat_dd'] = dflai.apply(lambda row: row['degrees N'] + row['min N']/60 + row['sec N']/3600, axis=1)
dflai['lon_dd'] = dflai.apply(lambda row: row['degrees E'] + row['min E']/60 + row['sec E']/3600, axis=1)


#%% extract pixel values from images at given coordinates
# 2021-06-29
imfile_path = "/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/5_Projects/2025Abisko/drone/orthomosaic4analysis/21_06_29_orthomosaic_georef_processed.tif"
ds = rasterio.open(imfile_path)
def sample_pixel_from_raster(ds, lon, lat) -> np.ndarray:
    """
    Given a rasterio dataset and lon/lat coordinates, return pixel band values as numpy array.
    """
    # Convert lon/lat to raster coordinates
    row, col = ds.index(lon, lat)
    vals = ds.read(window=((row, row), (col, col))).reshape(ds.count)
    return vals.astype(float)


#%%
# select relevant columns
# #!/usr/bin/env python3
# """
# /home/geofsn/GitHub/project-Abisko/src/lai/LAItrainer.py

# Utility to build a LAI prediction model by extracting pixel values from drone images
# using ground truth points. Supports:
#  - reading ground-truth CSV (columns: image, x, y OR image, lon, lat, lai)
#  - extracting pixel/band values from georeferenced rasters (rasterio) or plain images (PIL)
#  - combining data from multiple images
#  - validation: leave-one-out (per-sample) or leave-one-image-out
#  - trains and saves a scikit-learn regression model (RandomForest by default)

# Example CSV rows:
# image,x,y,lai
# img_01.tif,123,456,3.2

# image,lon,lat,lai
# img_01.tif,11.2345,59.1234,2.8
# """



# # Optional geospatial reading
# try:
#     HAS_RASTERIO = True
# except Exception:
#     HAS_RASTERIO = False



# def load_groundtruth(csv_path: str) -> pd.DataFrame:
#     df = pd.read_csv(csv_path)
#     # Standardize column names
#     df = df.rename(columns={c: c.strip().lower() for c in df.columns})
#     required = {'lai'}
#     if 'image' not in df.columns:
#         raise ValueError("Groundtruth CSV must contain an 'image' column with image filenames.")
#     if not ( {'x','y'}.issubset(df.columns) or {'lon','lat'}.issubset(df.columns) ):
#         raise ValueError("Groundtruth CSV must contain either pixel coords ('x','y') or geographic coords ('lon','lat').")
#     if 'lai' not in df.columns:
#         raise ValueError("Groundtruth CSV must contain 'lai' column (target).")
#     return df


# def open_image(path: str):
#     """
#     Return an object and metadata to allow sampling.
#     If rasterio is available and the file is a georeferenced raster, return rasterio dataset.
#     Otherwise return a PIL.Image.
#     """
#     if HAS_RASTERIO:
#         try:
#             ds = rasterio.open(path)
#             return ds
#         except Exception:
#             pass
#     # fallback to PIL
#     img = Image.open(path)
#     return img


# def sample_pixel_from_pil(img: Image.Image, x: int, y: int) -> np.ndarray:
#     # PIL uses (x, y) with origin at top-left, x horizontal, y vertical
#     w, h = img.size
#     if x < 0 or x >= w or y < 0 or y >= h:
#         raise ValueError(f"Pixel coordinate ({x},{y}) outside image bounds {img.size}")
#     px = img.getpixel((int(x), int(y)))
#     arr = np.array(px, dtype=float)
#     if arr.ndim == 0:
#         arr = arr.reshape(1)
#     return arr


# def sample_pixel_from_raster(ds, x=None, y=None, lon=None, lat=None) -> np.ndarray:
#     """
#     If lon/lat provided, use ds.index to get row/col or rasterio.sample for accuracy.
#     If x/y (pixel coords) provided, read window for that pixel.
#     Returns array of bands values (float).
#     """
#     if lon is not None and lat is not None:
#         # Convert lon/lat to raster coordinates if CRS differs assume same CRS if georeferenced
#         try:
#             # Transform world coords to row/col
#             row, col = ds.index(lon, lat)
#             # read band values at (row, col)
#             vals = ds.read(window=((row, row), (col, col))).reshape(ds.count)
#             return vals.astype(float)
#         except Exception:
#             # fallback to sample generator that accepts (x,y) pairs in dataset's CRS
#             coords = [(lon, lat)]
#             sampled = list(ds.sample(coords))
#             if len(sampled) == 0:
#                 raise ValueError("No sample returned from rasterio.sample")
#             return np.array(sampled[0], dtype=float)
#     elif x is not None and y is not None:
#         # rasterio uses row, col ordering; x,y are pixel coords (x is col)
#         col, row = int(x), int(y)
#         vals = ds.read(window=((row, row), (col, col))).reshape(ds.count)
#         return vals.astype(float)
#     else:
#         raise ValueError("Either x/y or lon/lat must be provided")


# def extract_features_for_row(row, images_dir: str):
#     """
#     Given a groundtruth row, open the corresponding image (image column) and extract pixel band values.
#     Returns a feature vector (1D numpy) or None if extraction failed.
#     """
#     image_fname = os.path.join(images_dir, str(row['image']))
#     if not os.path.exists(image_fname):
#         raise FileNotFoundError(f"Image file not found: {image_fname}")
#     img_obj = open_image(image_fname)
#     try:
#         if HAS_RASTERIO and hasattr(img_obj, 'read'):
#             # rasterio dataset
#             ds = img_obj
#             if 'lon' in row and 'lat' in row and not np.isnan(row['lon']) and not np.isnan(row['lat']):
#                 feat = sample_pixel_from_raster(ds, lon=float(row['lon']), lat=float(row['lat']))
#             else:
#                 feat = sample_pixel_from_raster(ds, x=int(row['x']), y=int(row['y']))
#             # optionally normalize or scale later
#             return feat
#         else:
#             # PIL image; only supports pixel coords x,y
#             if 'x' not in row or 'y' not in row:
#                 raise ValueError("PIL-based image sampling requires pixel coordinates 'x' and 'y' in groundtruth.")
#             feat = sample_pixel_from_pil(img_obj, int(row['x']), int(row['y']))
#             return feat
#     finally:
#         try:
#             if HAS_RASTERIO and hasattr(img_obj, 'close'):
#                 img_obj.close()
#         except Exception:
#             pass


# def build_dataset(gt_df: pd.DataFrame, images_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
#     features = []
#     targets = []
#     image_ids = []
#     missing = 0
#     for idx, row in gt_df.iterrows():
#         try:
#             feat = extract_features_for_row(row, images_dir)
#             features.append(np.asarray(feat, dtype=float).ravel())
#             targets.append(float(row['lai']))
#             image_ids.append(str(row['image']))
#         except Exception as e:
#             warnings.warn(f"Skipping row {idx} due to error: {e}")
#             missing += 1
#     if len(features) == 0:
#         raise RuntimeError("No features extracted. Check input CSV and images.")
#     X = np.vstack(features)
#     y = np.array(targets, dtype=float)
#     return X, y, image_ids


# def train_and_validate(X: np.ndarray, y: np.ndarray, image_ids: List[str], validation: str, model_type: str):
#     if model_type == 'rf':
#         base_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
#     elif model_type == 'ridge':
#         base_model = Ridge(alpha=1.0)
#     else:
#         raise ValueError("Unsupported model_type. Choose 'rf' or 'ridge'.")

#     if validation == 'leave_one_out':
#         loo = LeaveOneOut()
#         preds = np.zeros_like(y)
#         for train_idx, test_idx in loo.split(X):
#             model = base_model
#             model.fit(X[train_idx], y[train_idx])
#             preds[test_idx] = model.predict(X[test_idx])
#         rmse = np.sqrt(mean_squared_error(y, preds))
#         r2 = r2_score(y, preds)
#         print(f"LOO validation | samples: {len(y)} | RMSE: {rmse:.4f} | R2: {r2:.4f}")
#         # Fit final model on all data
#         final_model = base_model
#         final_model.fit(X, y)
#         return final_model, {'rmse': rmse, 'r2': r2}
#     elif validation == 'leave_one_image_out':
#         unique_images = list(dict.fromkeys(image_ids))  # preserve order
#         if len(unique_images) <= 1:
#             raise ValueError("Need multiple images for leave-one-image-out validation.")
#         preds = np.zeros_like(y)
#         for img in unique_images:
#             test_mask = np.array([iid == img for iid in image_ids])
#             train_mask = ~test_mask
#             if train_mask.sum() == 0:
#                 continue
#             model = base_model
#             model.fit(X[train_mask], y[train_mask])
#             preds[test_mask] = model.predict(X[test_mask])
#         rmse = np.sqrt(mean_squared_error(y, preds))
#         r2 = r2_score(y, preds)
#         print(f"Leave-one-image-out validation | images: {len(unique_images)} | RMSE: {rmse:.4f} | R2: {r2:.4f}")
#         final_model = base_model
#         final_model.fit(X, y)
#         return final_model, {'rmse': rmse, 'r2': r2}
#     else:
#         raise ValueError("Unsupported validation. Choose 'leave_one_out' or 'leave_one_image_out'.")


# def parse_args():
#     p = argparse.ArgumentParser(description="Build LAI prediction model from drone images and groundtruth points.")
#     p.add_argument('--groundtruth', '-g', required=True, help='CSV with groundtruth points (columns: image, x,y OR image, lon,lat, lai)')
#     p.add_argument('--images', '-i', required=True, help='Directory containing images referenced in CSV')
#     p.add_argument('--model-out', '-o', default='lai_model.joblib', help='Path to save trained model')
#     p.add_argument('--validation', '-v', default='leave_one_out', choices=['leave_one_out', 'leave_one_image_out'], help='Validation method')
#     p.add_argument('--model-type', default='rf', choices=['rf', 'ridge'], help='Model type to train')
#     return p.parse_args()


# def main():
#     args = parse_args()
#     print("Loading groundtruth...")
#     gt = load_groundtruth(args.groundtruth)
#     print(f"Found {len(gt)} groundtruth rows")
#     print("Extracting features from images...")
#     X, y, image_ids = build_dataset(gt, args.images)
#     print(f"Feature matrix: {X.shape}, Targets: {y.shape}")
#     print("Training and validating...")
#     model, metrics = train_and_validate(X, y, image_ids, args.validation, args.model_type)
#     print(f"Saving model to {args.model_out} ...")
#     dump({'model': model, 'feature_size': X.shape[1]}, args.model_out)
#     print("Done. Metrics:", metrics)


# if __name__ == '__main__':
#     main()
# %%
