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

img = glymur.Jp2k('../data/images/example.jp2')
img = img.read(rlevel=1)
img = img[4000:6000, 3000:img.shape[1] / 2, :]  # approximately symmetric

pca = decomp.PCA(1)
img = pca.fit_transform(img.reshape(-1, 3)).reshape(img.shape[:2])
img = (img - img.min()) / (img.max() - img.min())

binary = img < threshold_otsu(img)
distance = ndimage.distance_transform_edt(binary)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=binary)
markers = ndimage.label(local_maxi)[0]
labels = watershed(-distance, markers, mask=binary)
labeled_img = label2rgb(labels, img, bg_label=0)
