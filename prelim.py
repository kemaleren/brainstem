from __future__ import division

import glymur
import sklearn.decomposition as decomp
from skimage.filter import threshold_otsu
from skimage.morphology import label
from skimage.color import label2rgb
from scipy.stats import poisson
import numpy as np

from threshold import poisson_threshold

img = glymur.Jp2k('../data/images/example.jp2')
img = img.read(rlevel=1)
img = img[4000:6000, 3000:img.shape[1] / 2, :]  # approximately symmetric

pca = decomp.PCA(1)
img = pca.fit_transform(img.reshape(-1, 3)).reshape(img.shape[:2])
img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
threshold = poisson_threshold(img)
labels = label(img < threshold, background=0)
labeled_img = label2rgb(labels, img, bg_label=0)

