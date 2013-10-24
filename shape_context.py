from __future__ import division

import numpy as np
import scipy.stats
import scipy.spatial.distance as distance


def sample_points(img, n_points=100, multiplier=3):
    """Sample points along an edge.

    Uses rejection sampling to get points uniformly distributed along
    edges.

    Returns an array of shape ``(n_points, 2)``, where ``points[i,
    j]`` is in cartesian coordinates.

    """
    # FIXME: ensure in order along edge
    assert img.ndim == 2
    assert n_points > 0
    x, y = np.nonzero(img)
    points = np.hstack((y.reshape(-1, 1),
                        len(img) - x.reshape(-1, 1)))
    if len(points) < n_points:
        return points

    # sample n_points * multiplier
    n_sample_points = min(n_points * multiplier, len(points))
    idx = np.random.choice(len(points), n_sample_points)
    points = points[idx]
    n_total_points = len(points)

    # remove points closest to each other
    dists = distance.squareform(distance.pdist(points))
    dists = np.ma.masked_array(dists, mask=np.diag(np.ones(n_total_points)))
    while np.sqrt(dists.count() + n_total_points) > n_points:
        p1, p2 = np.unravel_index(dists.argmin(), dims=dists.shape)
        victim = np.random.choice((p1, p2))
        dists[victim] = np.ma.masked
        dists[:, victim] = np.ma.masked
    idx = np.unique(np.nonzero(-dists.mask)[0])        
    return points[idx]


def euclidean_dists_angles(points):
    """Returns symmetric ``dists`` and ``angles`` arrays."""
    # TODO: rotation invariance; compute angles relative to tangent
    n = len(points)
    dists = np.zeros((n, n))
    angles = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            dists[i, j] = dists[j, i] = np.linalg.norm(diff, ord=2)
            angles[i, j] = np.arctan2(*diff[::-1])
            angles[j, i] = np.arctan2(*(-diff)[::-1])
    return dists, angles


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

    # ensure distances are symmetric
    assert (dists.T == dists).all()

    n_points = dists.shape[0]

    r_array = np.logspace(0, 1, n_radial_bins, base=10) / 10.0
    r_array = np.hstack(([0], r_array))
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
        raise Exception('{} does not fit in any bin in {}.'
                        ' this should never happen.'.format(i, bins))

    for i in range(n_points):
        for j in range(n_points):
            if i == j:
                continue
            r_idx = get_idx(dists[i, j], r_array)
            theta_idx = get_idx(angles[i, j], theta_array)
            if r_idx != -1 and theta_idx != -1:
                result[i, r_idx, theta_idx] += 1
    # ensure all points were counted
    assert (result.reshape(n_points, -1).sum(axis=1) == (n_points - 1)).all()
    return result


def chi_square_distance(x, y):
    """Chi-square histogram distance.

    Ignores bins with no elements.

    """
    idx = (x + y != 0)
    x = x[idx]
    y = y[idx]
    x = x / x.max()
    y = y / y.max()
    num = np.power(x - y, 2)
    denom = x + y
    return (num / denom).sum() / 2


def shape_distance(a_descriptors, b_descriptors, penalty=0.3):
    """Computes the distance between two shapes.

    Uses dynamic programming to find best alignment of sampled points.
    Assumes point sequences are alignable (i.e. they do not need to be
    rotated.)

    """
    assert a_descriptors.ndim == 3
    assert b_descriptors.ndim == 3
    assert a_descriptors.shape[1:] == b_descriptors.shape[1:]

    n_rows = a_descriptors.shape[0]
    n_cols = b_descriptors.shape[0]

    a_descriptors = a_descriptors.reshape(n_rows, -1)
    b_descriptors = b_descriptors.reshape(n_rows, -1)

    table = np.zeros((n_rows, n_cols))

    d = lambda i, j: chi_square_distance(a_descriptors[i],
                                         b_descriptors[j])

    # initialize outer elements
    table[0, 0] = d(0, 0)

    for i in range(1, n_rows):
        match = i * penalty + d(i, 0)
        mismatch = table[i - 1, 0] + penalty
        table[i, 0] = min(match, mismatch)

    for j in range(1, n_cols):
        match = j * penalty + d(0, j)
        mismatch = table[0, j - 1] + penalty
        table[i, 0] = min(match, mismatch)

    # fill in the rest of the table
    for i in range(1, n_rows):
        for j in range(1, n_cols):
            match = table[i - 1, j - 1] + d(i, j)
            mismatch = min(table[i - 1, j],
                           table[i, j - 1]) + penalty
            table[i, j] = min(match, mismatch)

    # tracing optimal alignment is not necessary. we are just
    # interested in the final cost.
    return table[-1, -1]
            

def dists_to_affinities(dists, neighbors=10, alpha=0.27):
    """Compute an affinity matrix for a distance matrix."""
    affinities = np.zeros_like(dists)
    sorted_rows = np.sort(dists, axis=1)
    for i in range(dists.shape[0]):
        for j in range(i, dists.shape[1]):
            sigma = np.mean(sorted_rows[i, 1:neighbors],
                            sorted_rows[j, 1:neighbors])
            sim = scipy.stats.norm.pdf(dists[i, j], loc=0, scale=alpha * sigma)
            affinities[i, j] = sim
            affinities[j, i] = sim

    # normalize each row
    return affinities / affinities.sum(axis=1)


def graph_transduction(i, affinities, max_iters=5000):
    """Compute new affinities for a query based on graph transduction.

    The ``i``th element of ``affinities`` is the query; the rest are its
    candidate matches.

    """
    f = np.zeros((affinities.shape[0], 1))
    f[i] = 1
    for _ in range(max_iters):
        f = np.dot(affinities, f)
        f[i] = 1
    return f.ravel()
    

# TODO: for each shape, retrieve its nearest neighbors and only do
# graph transduction on them.

def compute_new_affinities(affinities):
    """Computes all new pairwise affinities by graph transduction."""
    result = list(graph_transduction(i, affinities) for i in range(affinities.shape[0]))
    # TODO: is the result symmetric?
    assert (affinities.T == affinities).all()
    return np.vstack(affinities)
