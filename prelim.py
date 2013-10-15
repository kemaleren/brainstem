from __future__ import division

import glymur

import numpy as np

from scipy.stats import poisson
from scipy import ndimage

import sklearn.decomposition as decomp

from skimage.filter import threshold_otsu
from skimage.color import label2rgb
from skimage.morphology import watershed
from skimage.feature import peak_local_max

jpimg = glymur.Jp2k('../data/images/example.jp2')
img = jpimg.read(rlevel=1)
img = img[4000:6000, 3000:img.shape[1] / 2, :]  # approximately symmetric

pca = decomp.PCA(1)
img = pca.fit_transform(img.reshape(-1, 3)).reshape(img.shape[:2])
img = (img - img.min()) / (img.max() - img.min())

# # global threshold and watershed
# binary = img < threshold_otsu(img)
# distance = ndimage.distance_transform_edt(binary)
# local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=binary)
# markers = ndimage.label(local_maxi)[0]
# labels = watershed(-distance, markers, mask=binary)
# labeled_img = label2rgb(labels, img, bg_label=0)

# local threshold and erosion / dilation
t_img = threshold_adaptive(img, 25, offset=.01)
b_img = m.binary_erosion(-t_img, np.ones((3, 3)))
d_img = m.binary_dilation(b_img, np.ones((3, 3)))
labels, _ = ndimage.label(d_img)
labeled = label2rgb(labels, img, bg_label=0)

# finding bounding box of brain in a slice
small_img = jpimg.read(rlevel=4)
blurred = gaussian_filter(small_img, 10)
ndimage.measurements.find_objects(blurred < threshold_otsu(blurred))
