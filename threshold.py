from __future__ import division

import numpy as np

def _error(t, hist):
    assert 1 <= t < 256

    prior_0 = hist[:t].sum()
    prior_1 = hist[t:].sum()

    a_0 = 0
    a_1 = t + 1

    b_0 = t
    b_1 = 256

    mu_0 = (hist[a_0 : b_0] * np.arange(a_0, b_0)).sum() / prior_0
    mu_1 = (hist[a_1 : b_1] * np.arange(a_1, b_1)).sum() / prior_1

    mu = prior_0 * mu_0 + prior_1 * mu_1

    return (mu -
            prior_0 * (np.log(prior_0) + mu_0 * np.log(mu_0)) -
            prior_1 * (np.log(prior_1) + mu_1 * np.log(mu_1)))


def poisson_threshold(img):
    assert img.dtype == np.uint8
    assert img.min() >= 0
    assert img.max() < 256
    hist, bins = np.histogram(img, bins=np.arange(0, 257), density=True)

    ts = np.arange(2, 255)
    errors = np.array(list(_error(t, hist) for t in ts))
    idx = np.argmin(errors)
    return ts[idx]
