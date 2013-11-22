"""Texture segmentation based on

Jain and Farrokhnia, "Unsupervised texture segmentation using Gabor filters" (1991)

"""

from __future__ import division

import numpy as np
from scipy import ndimage as nd

# from skimage.filter import gabor_kernel
from skimage.filter import gaussian_filter


def gabor_kernel(scale, orientation, Ul, Uh, n_scales, n_orientations, size, flag=True):
    # print "scale: {} orientation: {}".format(scale, orientation)
    # print "-------------------------"
    base = Uh / Ul
    a = np.power(base, 1.0 / (n_scales - 1))
    u0 = Uh / np.power(a, n_scales - scale)
    Uvar = (a - 1.0) * u0 / ((a + 1.0) * np.sqrt(2.0 * np.log(2.0)))
    z = -2.0 * np.log(2.0) * (Uvar * Uvar) / u0
    Vvar = np.tan(np.pi / ( 2 * n_orientations)) * (u0 + z) / np.sqrt(2.0 * np.log(2.0) - z * z / (Uvar * Uvar))

    Xvar = 1.0 / (2.0 * np.pi * Uvar)
    Yvar = 1.0 / (2.0 * np.pi * Vvar)

    t1 = np.cos(np.pi / n_orientations * (orientation - 1.0))
    t2 = np.sin(np.pi / n_orientations * (orientation - 1.0))

    # print "base: {}".format(base)
    # print "a: {}".format(a)
    # print "u0: {}".format(u0)
    # print "Uvar: {}".format(Uvar)
    # print "z: {}".format(z)
    # print "Vvar: {}".format(Vvar)
    # print "Xvar: {}".format(Xvar)
    # print "Yvar: {}".format(Yvar)
    # print "t1: {}".format(t1)
    # print "t2: {}".format(t2)
    # print ""

    side = int((size - 1) / 2)

    result = np.zeros((size, size), dtype=np.complex)
    
    for x in range(size):
        for y in range(size):
            X = (x - side) * t1 + (y - side) * t2
            Y = -(x - side) * t2 + (y - side) * t1
            G = 1.0 / (2.0 * np.pi * Xvar * Yvar) * np.power(a, n_scales - scale) * np.exp(-0.5 * ((X * X) / (Xvar * Xvar) + (Y * Y) / (Yvar * Yvar)))
            result[x, y] = G * np.cos(2.0 * np.pi * u0 * X) + 1j * G * np.sin(2.0 * np.pi * u0 * X)

    if flag:
        m = 0
        for x in range(2 * side + 1):
            for y in range(2 * side + 1):
                m += np.real(result[x, y])

        m /= np.power(2.0 * side + 1, 2.0)
        
        for x in range(2 * side + 1):
            for y in range(2 * side + 1):
                result[x, y] -= m
    return result, u0


def make_filter_bank(Ul, Uh, n_scales, n_orientations, size, flag):
    """prepare filter bank of kernels"""
    result = list(gabor_kernel(s, n, Ul, Uh, n_scales, n_orientations, size, flag)
                  for s in range(1, n_scales + 1)
                  for n in range(1, n_orientations + 1))
    kernels, freqs = zip(*result)
    return kernels, np.array(freqs)


def filter_image(image, kernels, frequencies, r2=0.95, select=True):
    """Computes all convolutions and discards some filtered images.

    Returns filtered images with the largest energies so that the
    coefficient of determiniation is >= ``r2``.

    """
    # TODO: faster in fourier domain?
    filtered = np.dstack(nd.convolve(image, kernel, mode='wrap')
                         for kernel in kernels)
    if not select:
        return filtered, frequencies
    energies = filtered.sum(axis=0).sum(axis=0)

    # sort from largest to smallest energy
    idx = np.argsort(energies)[::-1]
    filtered = filtered[:, :, idx]
    energies = energies[idx]
    total_energy = energies.sum()

    r2s = np.cumsum(energies) / energies.sum()
    k = np.searchsorted(r2s, r2)
    n_start = filtered.shape[2]
    return filtered[:, :, :k], frequencies[idx][:k]


def compute_features(filtered, frequencies,
                     proportion=0.5,
                     alpha=0.25):
    """Compute features for each filtered image.

    ``frequencies[i]`` is the center frequency of the Gabor filter
    that generated ``filtered[i]``.

    """
    # TODO: is this really what the paper means in formula 6?
    nonlinear = np.tanh(alpha * filtered)
    ncols = filtered.shape[1]
    sigmas = proportion * ncols * np.array(frequencies)
    features = np.dstack(gaussian_filter(nonlinear[:, :, i], sigmas[i])
                         for i in range(len(sigmas)))
    return features


def add_coordinates(features, spatial_importance=1.0):
    """Adds coordinates to each feature vector and normalizes."""
    n_rows, n_cols = features.shape[:2]
    coords = np.mgrid[:n_rows, :n_cols].swapaxes(0, 2).swapaxes(0, 1)
    features = np.dstack((features, coords))
    n_feats = features.shape[2]

    means = np.array(list(features[:, :, i].mean() for i in range(n_feats)))
    stds = np.array(list(features[:, :, i].std(ddof=1) for i in range(n_feats)))

    means = means.reshape(1, 1, -1)
    stds = stds.reshape(1, 1, -1)

    features = (features - means) / stds
    features[:, :, -2:] *= spatial_importance
    return features


def _get_freqs(img):
    n_cols = img.shape[1]
    next_pow2 = 2 ** int(np.ceil(np.log2(n_cols)))
    min_freq = next_pow2 / 4
    n_freqs = int(np.log2(min_freq)) + 2
    return list(np.sqrt(2) / float(2 ** i) for i in range(n_freqs))


def segment_textures(img, model):
    # TODO: these filter parameters are not correct
    # frequencies = _get_freqs(img)
    # thetas = np.deg2rad([0, 45, 90, 135])
    kernels, all_freqs = make_filter_bank(0.1, 0.4, 3, 4, 60, False)
    kernels = list(np.real(k) for k in kernels)
    filtered, all_freqs = filter_image(img, kernels, all_freqs)
    features = compute_features(filtered, all_freqs)
    features = add_coordinates(features)
    n_feats = features.shape[-1]
    model.fit(features.reshape(-1, n_feats))
    return model.labels_.reshape(img.shape)
