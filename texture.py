"""Texture segmentation based on

Jain and Farrokhnia, "Unsupervised texture segmentation using Gabor filters" (1991)

"""

from __future__ import division

import numpy as np
from scipy.signal import fftconvolve

from skimage.filter import gabor_kernel
from skimage.filter import gaussian_filter

from sklearn.decomposition import PCA


def _compute_sigmas(frequency, freq_band=1, angular_band=np.deg2rad(45)):
    sigma_x = np.sqrt(np.log(2)) * (2 ** freq_band + 1) / (np.sqrt(2) * np.pi * frequency * (2 ** freq_band - 1))
    sigma_y = np.sqrt(np.log(2)) / (np.sqrt(2) * np.pi * frequency * np.tan(angular_band / 2))
    return sigma_x, sigma_y


def make_filter_bank(frequencies, thetas, real=True):
    """prepare filter bank of kernels"""
    # TODO: set MTF of each filter at (u, v) to 0
    kernels = []
    all_freqs = []
    for frequency in frequencies:
        sigma_x, sigma_y = _compute_sigmas(frequency)
        for theta in thetas:
            kernel = gabor_kernel(frequency, theta=theta,
                                  bandwidth=1)
            kernels.append(kernel)
            all_freqs.append(frequency)
    if real:
        kernels = list(np.real(k) for k in kernels)
    return kernels, np.array(all_freqs)


def filter_image(image, kernels, frequencies, crop=True, select=True, r2=0.95):
    """Computes all convolutions and discards some filtered images.

    Returns filtered images with the largest energies so that the
    coefficient of determiniation is >= ``r2``.

    """
    filtered = np.dstack(fftconvolve(image, kernel, 'same')
                         for kernel in kernels)

    if crop:
        x = max(k.shape[0] for k in kernels)
        y = max(k.shape[1] for k in kernels)
        x = int(np.ceil(x / 2))
        y = int(np.ceil(y / 2))
        filtered = filtered[x:-x, y:-y]

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

    # paper says proportion * n_cols / frequency, but remember that
    # their frequency is in cycles per image width. our frequency is
    # in cycles per pixel, so we just need to take the inverse.
    sigmas = proportion * (1.0 / np.array(frequencies))
    features = np.dstack(gaussian_filter(nonlinear[:, :, i], sigmas[i])
                         for i in range(len(sigmas)))
    return features


def add_coordinates(features, spatial_importance=1.0):
    """Adds coordinates to each feature vector and standardizes each feature."""
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


def get_freqs(img):
    """Compute the appropriate frequencies for an image of the given shape.

    Frequencies are given in cycles/pixel.

    """
    n_cols = img.shape[1]
    next_pow2 = 2 ** int(np.ceil(np.log2(n_cols)))
    min_freq = next_pow2 / 4
    n_freqs = int(np.log2(min_freq)) + 2

    # note: paper gives frequency in cycles per image width.
    # we need cycles per pixel, so divide by image width
    frequencies =  list((np.sqrt(2) * float(2 ** i)) / n_cols
                        for i in range(n_freqs))
    return frequencies


def segment_textures(img, model, freqs=None, thetas=None, n_thetas=4, select=True, k=4, coord=1):
    """Segments textures using Gabor filters and k-means."""
    if freqs is None:
        freqs = get_freqs(img)[-5:]
    if thetas is None:
        thetas = np.deg2rad(np.arange(0, 180, 180.0 / n_thetas))
    kernels, all_freqs = make_filter_bank(freqs, thetas)
    filtered, all_freqs = filter_image(img, kernels, all_freqs, select=select)
    features = compute_features(filtered, all_freqs)
    features = add_coordinates(features, spatial_importance=coord)
    n_feats = features.shape[-1]
    X = features.reshape(-1, n_feats)
    pca = PCA(k)
    X = pca.fit_transform(X)
    model.fit(X)
    return model.labels_.reshape(img.shape)


def directionality_filter(img, freqs=None, thetas=None, n_thetas=18):
    """
    Finds the maximum filter response for each pixel.

    Returns the maximum filter response and the angle of maximum response.

    """
    if freqs is None:
        freqs = get_freqs(img)[-5:-2]
    if thetas is None:
        thetas = np.deg2rad(np.arange(0, 180, 180.0 / n_thetas))

    freqs = get_freqs(img)
    thetas = np.deg2rad(np.arange(0, 180, 10))

    kernels, all_freqs = make_filter_bank(freqs, thetas)
    filtered, all_freqs = filter_image(img, kernels, all_freqs, select=False)
    f2 = np.power(filtered, 2)

    n_thetas = len(thetas)
    f2_angles = np.dstack(f2[:, :, i::n_thetas].sum(axis=2)
                          for i in range(n_thetas))

    max_angle_idx = np.argmax(f2_angles, axis=2)
    x, y = np.indices(max_angle_idx.shape)
    f2_maxes = f2[x, y, max_angle_idx]
    magnitude = f2_maxes / f2.mean(axis=2)

    max_thetas = np.rad2deg(np.array(thetas))[max_angle_idx]
    return magnitude, max_thetas


def scale(arr):
    """scale array to [0, 1]"""
    return (arr - arr.min()) / arr.max()


def make_hsv(magnitude, angle):
    """Convert the result of ``directionality_filter`` to an HSV image"""
    magnitude = scale(magnitude)
    angle = scale(angle)
    h = angle
    s = magnitude
    v = np.ones(h.shape)
    return np.dstack([h, s, v])
