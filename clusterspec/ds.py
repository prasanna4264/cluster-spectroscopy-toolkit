"""Dressler & Shectman (1988) test for 3D substructure.

For each member, compare the mean velocity and dispersion of it and its nearest neighbours on
the sky with the cluster values:

    delta_i^2 = (N_loc / sigma^2) [ (vbar_loc - vbar)^2 + (sigma_loc - sigma)^2 ]

and sum Delta = sum_i delta_i. Significance comes from Monte Carlo: shuffle velocities among
the fixed positions and count how often the shuffled Delta exceeds the observed one.

Conventions here (change them if your reference differs):
  * local group = the galaxy itself + n_nn nearest neighbours, n_nn = max(8, sqrt(N))
    (so N_loc = n_nn + 1);
  * local mean = plain mean of the group's velocities; local and global dispersions are
    biweight scales; global mean = biweight location;
  * the statistic reported is Delta / N;
  * shuffling permutes (velocity, redshift) pairs together.
"""
import numpy as np
from scipy.spatial import cKDTree
from .stats import biweight_location_v, biweight_scale_v
from .cosmo import C_KMS


def unit_vectors(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.c_[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]


def default_n_neighbours(N):
    return max(8, int(np.sqrt(N)))


def neighbour_table(ra, dec, n_nn=None):
    """(N, n_nn + 1) indices: each galaxy and its n_nn nearest neighbours on the sky.
    Ties (identical positions) are broken by index order."""
    X = unit_vectors(np.asarray(ra, float), np.asarray(dec, float))
    n_nn = default_n_neighbours(len(X)) if n_nn is None else n_nn
    _, idx = cKDTree(X).query(X, k=n_nn + 1)
    # make sure each row starts with the galaxy itself (matters only for duplicate positions)
    for i in range(len(X)):
        if idx[i, 0] != i:
            row = [i] + [j for j in idx[i] if j != i]
            idx[i] = row[:n_nn + 1]
    return idx


def ds_delta(v, z, idx, sigma_global, per_galaxy=False):
    """Delta / N for velocities v (km/s) and redshifts z. Leading axes of v and z are treated
    as independent realisations (e.g. Monte Carlo shuffles). With per_galaxy=True also
    returns delta_i."""
    v, z = np.asarray(v, float), np.asarray(z, float)
    n_loc = idx.shape[1]
    vg = biweight_location_v(v)[..., None]
    vl = v[..., idx].mean(-1)
    zl = z[..., idx]
    sl = C_KMS * biweight_scale_v(zl) / (1 + biweight_location_v(zl))
    d = np.sqrt((n_loc / sigma_global ** 2) * ((vl - vg) ** 2 + (sl - sigma_global) ** 2))
    D = d.sum(-1) / v.shape[-1]
    return (D, d) if per_galaxy else D


def ds_test(v, z, ra, dec, sigma_global, n_mc=1000, n_nn=None, seed=0, chunk=250, idx=None):
    """Run the DS test. Returns dict(delta_obs, delta_mc, p_value, delta_i, idx).

    p_value is the fraction of shuffles with Delta/N >= observed; with n_mc shuffles the
    smallest resolvable p is 1/n_mc."""
    v, z = np.asarray(v, float), np.asarray(z, float)
    idx = neighbour_table(ra, dec, n_nn) if idx is None else idx
    obs, d_i = ds_delta(v, z, idx, sigma_global, per_galaxy=True)
    rng = np.random.default_rng(seed)
    perms = np.array([rng.permutation(v.size) for _ in range(n_mc)])
    mc = np.concatenate([ds_delta(v[p], z[p], idx, sigma_global)
                         for p in np.array_split(perms, max(1, n_mc // chunk))])
    return dict(delta_obs=float(obs), delta_mc=mc, p_value=float(np.mean(mc >= obs)),
                delta_i=d_i, idx=idx)
