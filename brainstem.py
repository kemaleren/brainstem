from __future__ import division

import os
from cPickle import dump, load

import glymur

import numpy as np

from scipy.stats import poisson
from scipy import ndimage
from scipy import misc

import sklearn.decomposition as decomp

from skimage.filter import threshold_otsu, threshold_adaptive
from skimage.morphology import binary_dilation, binary_erosion
from skimage.color import label2rgb
from skimage.morphology import watershed
from skimage.feature import peak_local_max

# a simple cache for grayscale images at different resolutions.
USE_CACHE = True
DATA_DIR = os.path.expanduser("~/devel/data/images/PMD1305_N")
CACHE_DIR = os.path.expanduser("~/devel/data/cache")
CACHE_PATH = os.path.join(CACHE_DIR, 'cache.pkl')

if USE_CACHE:
    if not os.path.exists(CACHE_DIR):
        os.mkdir(CACHE_DIR)
    if not os.path.isdir(CACHE_DIR):
        raise Exception('path is not a directory: {}'.format(CACHE_DIR))
    try:
        CACHE = load(file(CACHE_PATH, 'rb'))
    except IOError:
        CACHE = {}


def get_filenames():
    """returns a list of all jp2 images in the data directory"""
    filenames = os.listdir(DATA_DIR)
    filenames = list(n for n in filenames if os.path.splitext(n)[1] == '.jp2')
    filenames = sorted(filenames, key=lambda n: n.split('_')[-1])
    return filenames


def _read_jp2_img(filename, rlevel):
    """read a jp2 image at the given level of resolution.

    rlevel = 0 is highest resolution.
    rlevel = -1 is a shortcut for the lowest resolution.

    """
    jpimg = glymur.Jp2k(os.path.join(DATA_DIR, filename))
    return jpimg.read(rlevel=rlevel)


def read_img(filename, rlevel):
    """get an image, using the cache if available."""
    try:
        fname = CACHE[(filename, rlevel)]
        return misc.imread(fname)
    except:
        img = _read_jp2_img(filename, rlevel)
        if USE_CACHE:
            cache_file = os.path.join(CACHE_DIR, "{}_rlevel_{}.tif".format(filename, rlevel))
            misc.imsave(cache_file, img)
            CACHE[(filename, rlevel)] = cache_file
            dump(CACHE, open(CACHE_PATH, 'wb'))
        return img


def make_grey(img):
    """convert a color image to grayscale using PCA"""
    pca = decomp.PCA(1)
    img = pca.fit_transform(img.reshape(-1, 3)).reshape(img.shape[:2])
    return (img - img.min()) / (img.max() - img.min())


def get_cutout(filename):
    """read an image, cropping out the background"""
    # find bounding box of brain in a slice
    small_img = read_img(filename, rlevel=4)
    small_img = make_grey(small_img)
    blurred = ndimage.gaussian_filter(small_img, 10)
    slc = ndimage.measurements.find_objects(blurred < threshold_otsu(blurred))[0]
    x_slc = slice(slc[0].start * 2 ** 3, slc[0].stop * 2 ** 3)
    y_slc = slice(slc[1].start * 2 ** 3, slc[1].stop * 2 ** 3)
    img = read_img(filename, rlevel=1)
    return img[x_slc, y_slc]


def random_image_sample(img, scale=5):
    """extract a random subset of the image.

    The result is ``scale`` times smaller in each dimension.

    """
    x_shape = int(np.floor(img.shape[0] / scale))
    y_shape = int(np.floor(img.shape[1] / scale))
    x_start = np.random.randint(0, img.shape[0] - x_shape)
    y_start = np.random.randint(0, img.shape[1] - y_shape)

    return img[x_start : x_start + x_shape,
               y_start : y_start + y_shape]


def segment_cells(img, rgb=False):
    """label the cells in an image and return the labeled image.

    If the ``rgb`` parameter is True, returns
    ``skimage.color.label2rgb()`` on the result, which is convenient
    for visualization.

    """
    # # global threshold and watershed
    # binary = img < threshold_otsu(img)
    # distance = ndimage.distance_transform_edt(binary)
    # local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=binary)
    # markers = ndimage.label(local_maxi)[0]
    # labels = watershed(-distance, markers, mask=binary)

    # local threshold and erosion / dilation
    img = make_grey(img)
    t_img = threshold_adaptive(img, 25, offset=.01)
    b_img = binary_erosion(-t_img, np.ones((3, 3)))
    d_img = binary_dilation(b_img, np.ones((3, 3)))
    labels, _ = ndimage.label(d_img)

    if rgb:
        return label2rgb(labels, img, bg_label=0)
    return labels

