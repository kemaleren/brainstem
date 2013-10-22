from __future__ import division

import numpy as np

def shape_context(dists, angles, n_radial_bins=5, n_polar_bins=12):
    """Compute shape context descriptors for all given points.

    If ``dists`` and ``angles`` are Euclidean, this corresponds to the
    original shape context (Belongie, Malik, and Puzicha, 2002). If
    they are the inner-distance and inner angle, this corresponds to
    the inner-distance shape context (Ling and Jacobs, 2007).

    Parameters
    ----------
    dists : (N, N) ndarray
        ``dists[i, j]`` is the distance between points ``i`` and ``j``.

    angles : (N, N) ndarray
        ``angles[i, j]`` is the distance between points ``i`` and ``j``.

    n_radial_bins : int
        number of radial bins in histogram

    n_polar_bins : int
        number of polar bins in histogram
    
    Returns
    -------
    shape_contexts : ndarray
        The shape context descriptor for each point. Has shape
        ``(n_points, radial_bins, polar_bins)``.

    """
    assert dists.shape[0] == dists.shape[1]
    assert dists.ndim == 2
    assert dists.shape == angles.shape

    # ensure distances and angles are symmetric
    assert (dists.transpose(1, 0, 2) == dists).all()
    assert (angles.transpose(1, 0, 2) == angles).all()

    n_points = dists.shape[0]

    r_array = np.logspace(0, 1, n_radial_bins + 1, base=10) / 10.0
    theta_array = np.linspace(-np.pi, np.pi, n_polar_bins + 1)
    result = np.zeros((n_points, n_radial_bins, n_polar_bins),
                      dtype=np.int)

    # normalize distances
    dists = dists / dists.max()

    # TODO: do in Cython
    def get_idx(i, bins):
        assert bins.ndim == 1
        for idx in range(0, bins.size - 1):
            if bins[idx] <= i < bins[idx + 1]:
                return idx
        if i == bins[idx + 1]:
            return idx
        return -1

    for i in range(n_points):
        for j in range(i + 1, n_points):
            r_idx = get_idx(dists[i, j], r_array)
            theta_idx = get_idx(angles[i, j], theta_array)
            if r_idx != -1 and theta_idx != -1:
                result[r_idx, theta_idx] += 1
    return result
            
