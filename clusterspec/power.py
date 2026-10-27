"""How sensitive is the DS test for a given cluster? Mock velocities on the real positions.

The detection power of the DS test depends strongly on N and on the spatial sampling, so a
non-detection in a small cluster says little. ``ds_power`` keeps the observed sky positions
and draws velocities for:

  null    all galaxies Gaussian with dispersion sigma -> false-positive rate (should be ~alpha)
  merger  a compact subgroup (the n_sub galaxies nearest a random galaxy, optionally chosen
          only among "core" galaxies) offset by dv * sigma along the line of sight, with
          internal dispersion sub_sigma * sigma; everyone else Gaussian.

and returns the fraction of mocks with p <= alpha.
"""
import numpy as np
from .ds import neighbour_table, ds_delta, unit_vectors, default_n_neighbours
from .stats import biweight_location, biweight_scale
from .cosmo import C_KMS


def _p_value(v, z, idx, n_mc, rng):
    sg = C_KMS * biweight_scale(z) / (1 + biweight_location(z))
    obs = ds_delta(v, z, idx, sg)
    perms = np.array([rng.permutation(v.size) for _ in range(n_mc)])
    mc = ds_delta(v[perms], z[perms], idx, sg)
    return np.mean(mc >= obs)


def ds_power(ra, dec, sigma, z_cl, frac_sub=0.25, dv=1.5, sub_sigma=0.5, core=None,
             n_mock=200, n_mc=300, alpha=0.05, seed=0, n_nn=None):
    """Detection rate of the DS test for mock clusters on the given positions.

    frac_sub : subgroup size as a fraction of N (or of the core, if ``core`` is given);
               0 gives the null (false-positive) rate.
    dv       : subgroup velocity offset in units of sigma (sign random).
    core     : optional boolean mask; subgroups are built only from these galaxies.
    Returns the fraction of n_mock mocks with p <= alpha."""
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    N = ra.size
    rng = np.random.default_rng(seed)
    idx = neighbour_table(ra, dec, default_n_neighbours(N) if n_nn is None else n_nn)
    X = unit_vectors(ra, dec)
    pool = np.arange(N) if core is None else np.where(core)[0]
    n_sub = max(3, int(round(frac_sub * pool.size))) if frac_sub > 0 else 0
    det = 0
    for _ in range(n_mock):
        v = rng.normal(0, sigma, N)
        if n_sub:
            s = rng.choice(pool)
            grp = pool[np.argsort(((X[pool] - X[s]) ** 2).sum(1))[:n_sub]]
            v[grp] = rng.normal(dv * sigma * rng.choice([-1, 1]), sub_sigma * sigma, n_sub)
        z = z_cl + v * (1 + z_cl) / C_KMS
        det += _p_value(v, z, idx, n_mc, rng) <= alpha
    return det / n_mock
