"""Texture segmentation based on

Jain and Farrokhnia, "Unsupervised texture segmentation using Gabor filters" (1991)

"""

from __future__ import division

import numpy as np
from scipy import ndimage as nd

from skimage.filter import gabor_kernel
from skimage.filter import gaussian_filter


def _compute_sigmas(frequency, freq_band=1, angular_band=np.deg2rad(45)):
    sigma_x = np.sqrt(np.log(2)) * (2 ** freq_band + 1) / (np.sqrt(2) * np.pi * frequency * (2 ** freq_band - 1))
    sigma_y = np.sqrt(np.log(2)) / (np.sqrt(2) * np.pi * frequency * np.tan(angular_band / 2))
    return sigma_x, sigma_y


def make_filter_bank(frequencies, thetas):
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
    return kernels, np.array(all_freqs)


def filter_image(image, kernels, frequencies, r2=0.95, select=True):
    """Computes all convolutions and discards some filtered images.

    Returns filtered images with the largest energies so that the
    coefficient of determiniation is >= ``r2``.

    """
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

    # paper says proportion * n_cols / frequency, but remember that
    # their frequency is in cycles per image width. our frequency is
    # in cycles per pixel, so we just need to take the inverse.
    sigmas = proportion * (1.0 / np.array(frequencies))
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


def get_freqs(img):
    n_cols = img.shape[1]
    next_pow2 = 2 ** int(np.ceil(np.log2(n_cols)))
    min_freq = next_pow2 / 4
    n_freqs = int(np.log2(min_freq)) + 2

    # note: paper gives frequency in cycles per image width.
    # we need cycles per pixel, so divide by image width
    frequencies =  list((np.sqrt(2) * float(2 ** i)) / n_cols
                        for i in range(n_freqs))

    # only keep 5 highest frequencies
    return frequencies[-5:]


def segment_textures(img, model):
    frequencies = get_freqs(img)
    thetas = np.deg2rad([0, 45, 90, 135])
    kernels, all_freqs = make_filter_bank(frequencies, thetas)
    kernels = list(np.real(k) for k in kernels)
    filtered, all_freqs = filter_image(img, kernels, all_freqs, select=True)
    features = compute_features(filtered, all_freqs)
    features = add_coordinates(features)
    n_feats = features.shape[-1]
    model.fit(features.reshape(-1, n_feats))
    return model.labels_.reshape(img.shape)
