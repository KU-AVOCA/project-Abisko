#%%
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import median, gaussian, threshold_otsu, sobel
from skimage.morphology import binary_erosion
from skimage import morphology

#%%
image_path = '/mnt/i/SCIENCE-IGN-ALL/AVOCA_Group/2_Shared_folders/1_Data/1_Abisko/9_RGB_Close-up/1_CHMB/2_2022/20220622_G1.JPG'  # Replace with the actual path to your image
orig = imread(image_path,as_gray=True)
img = orig.copy()
img[img<1] = 0

gauss = gaussian(img, sigma=3)

SE = np.ones((7,7))
med = median(gauss)

edges = sobel(med)

thresh = threshold_otsu(edges)
binary = edges > thresh

SE2 = np.ones((3,3))
result = binary_erosion(binary)


result = morphology.remove_small_objects(result, min_size=50)
y, x = np.where(result)

if len(x) > 0:
    min_x, min_y = np.min(x), np.min(y)
    max_x, max_y = np.max(x), np.max(y)
    print(f"Bounding Box: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")
else:
    print("No objects found in the binary image.")
    min_x, min_y, max_x, max_y = None, None, None, None

plt.subplot(121)
plt.imshow(orig, cmap='gray')
plt.axis('off')
plt.subplot(122)
plt.imshow(result, cmap='gray')
plt.axis('off')
plt.show()
# %%
