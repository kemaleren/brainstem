from __future__ import division

from collections import defaultdict
import itertools

import numpy as np
import scipy.stats
from scipy import ndimage

from skimage.segmentation import find_boundaries
from skimage.morphology import skeletonize


def pixel_graph(img):
    """ Create an 8-way pixel connectivity graph for a binary image."""
    m, n = img.shape
    adj = defaultdict(set)
    # TODO: inelegant; repeated effort
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            for imod in (-1, 0, 1):
                for jmod in (-1, 0, 1):
                    if i + imod < 0 or j + jmod < 0:
                        continue
                    if i + imod >= m or j + jmod >= n:
                        continue
                    if imod == jmod == 0:
                        continue
                    if img[i, j] and img[i + imod, j + jmod]:
                        adj[i, j].add((i + imod, j + jmod))
                        adj[i + imod, j + jmod].add((i, j))
    return adj


def _sample_single_contour(img, n_points):
    """Samples pixels, in order, along a contour.

    Parameters
    ----------
    img : np.ndarray
        A binary image with nonzero elements along a connected
        contour.
    n_points : the number of points to return

    Returns
    -------
    points : list of tuples
        An ordered list of (x, y) points, starting from the point
        closest to the origin.

    """
    # Right now, just does a depth-first search. This is not optimal
    # because backtracking can put some pixels very far out of
    # order. a better approach would be to find a locally stable
    # sorting. However, this seems to work well enough for now.
    graph = pixel_graph(img)
    visited = set()
    unvisited = set(graph.keys())
    stacked = set()
    stack = []
    order = []
    while unvisited:
        assert len(visited) + len(unvisited) == len(graph)
        try:
            node = stack.pop()
        except:
            # TODO: this is not actually closest to the origin
            node = min(unvisited)
        assert not node in visited
        order.append(node)
        visited.add(node)
        unvisited.remove(node)
        neighbors = graph[node]
        for n in neighbors - stacked - visited:
            stack.append(n)
            stacked.add(n)
        assert len(visited) + len(unvisited) == len(graph)
        assert len(visited & unvisited) == 0
    assert len(order) == len(graph)
    stride = int(np.ceil(len(order) / n_points))
    return order[::stride]


def sample_points(img, n_points=100):
    """Sample points along edges in a binary image.

    Returns an array of shape ``(n_points, 2)`` in image coordinates.

    If there are several disconnected contours, they are sampled
    seperately and appended in order of their minimum distance to the
    origin of ``img`` in NumPy array coordinates.

    """
    # FIXME: what if contour crosses itself? for example: an infinity
    # symbol?
    assert img.ndim == 2
    assert n_points > 0

    boundaries = skeletonize(find_boundaries(img))

    # reorder along curves; account for holes and disconnected lines
    # with connected components.
    labels, n_labels = ndimage.label(boundaries, structure=np.ones((3, 3)))
    n_pixels = labels.sum()
    curve_pixels = list((labels == lab + 1).sum() for lab in range(n_labels))
    curve_n_points = list(int(np.ceil((p / n_pixels) * n_points))
                          for p in curve_pixels)

    # sample a linear subset of each connected curve
    samples = list(_sample_single_contour(labels == lab + 1, n)
                   for lab, n in enumerate(curve_n_points))

    # append them together. They should be in order, because
    # ndimage.label() labels in order.
    points = list(itertools.chain(*samples))
    return np.vstack(points)


def euclidean_dists_angles(points):
    """Returns symmetric pairwise ``dists`` and ``angles`` arrays."""
    # TODO: rotation invariance; compute angles relative to tangent
    n = len(points)
    dists = scipy.spatial.distance.pdist(points, 'euclidean')
    dists = scipy.spatial.distance.squareform(dists)
    # TODO: can we do angles with scipy too?
    angles = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
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
    """Chi-square histogram distance between vectors ``x`` and ``y``.

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

    The distance is defined as the minimal cost of aligning an ordered
    sequence of shape context descriptors along their contours. For
    more information, see Ling and Jacobs, 2007.

    Uses dynamic programming to find best alignment of sampled points.

    """
    # FIXME: Assumes the sequences start from the correct position
    # TODO: this could probably be optimized.

    assert a_descriptors.ndim == 3
    assert b_descriptors.ndim == 3
    assert a_descriptors.shape[1:] == b_descriptors.shape[1:]

    n_rows = a_descriptors.shape[0]
    n_cols = b_descriptors.shape[0]

    a_descriptors = a_descriptors.reshape(n_rows, -1)
    b_descriptors = b_descriptors.reshape(n_cols, -1)

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


def full_shape_distance(img1, img2):
    """A convenience function to compute the distance between two binary images."""
    points1 = sample_points(img1)
    dists1, angles1 = euclidean_dists_angles(points1)
    descriptors1 = shape_context(dists1, angles1)

    points2 = sample_points(img2)
    dists2, angles2 = euclidean_dists_angles(points2)
    descriptors2 = shape_context(dists2, angles2)

    return shape_distance(descriptors1, descriptors2)


def dists_to_affinities(dists, neighbors=10, alpha=0.27):
    """Compute an affinity matrix for a given distance matrix."""
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

    Method as described in "Learning context-sensitive shape
    similarity by graph transduction." by Bai, Yang, et. al. (2010).

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
