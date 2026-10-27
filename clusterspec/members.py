"""Cluster membership.

Three ways to pick members, all starting from a broad velocity window around a seed redshift:

* ``velocity``: iterative 3-sigma biweight clipping in redshift only.
* ``fixed``:    keep galaxies within a fixed projected radius, then clip.
* ``r200``:     iterate radius -> clip -> sigma_v -> M200 -> R200 -> radius until R200 stops
                changing (damped, with limit-cycle detection).

Velocity-only membership is the classical choice, but in widely surveyed fields it keeps
galaxies several Mpc from the cluster (infall regions, neighbouring groups), which can drive
substructure tests. Comparing the three is a cheap robustness check.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .stats import biweight_location, biweight_scale
from .dynamics import dynamics, m200_from_sigma, r200_from_m200, projected_distance
from .cosmo import PLANCK18, C_KMS


def velocity_window(z, z_seed, dv_max=5000.0):
    """Boolean mask of galaxies within +/- dv_max km/s (rest frame) of z_seed."""
    z = np.asarray(z, float)
    return C_KMS * np.abs(z - z_seed) / (1 + z_seed) < dv_max


def sigma_clip_members(z, nsig=3.0, max_iter=100):
    """Iterative biweight clipping: drop galaxies outside location +/- nsig * scale (in z)
    until nothing changes. Returns a boolean mask into ``z``."""
    z = np.asarray(z, float)
    keep = np.ones(z.size, bool)
    for _ in range(max_iter):
        loc, scl = biweight_location(z[keep]), biweight_scale(z[keep])
        new = keep & (z >= loc - nsig * scl) & (z <= loc + nsig * scl)
        if new.sum() == keep.sum():
            break
        keep = new
    return keep


def seed_redshift(z, ra=None, dec=None, zmin=0.05, zmax=2.0, bin_width=0.01, smooth=0.02,
                  dv=5000.0):
    """Estimate a cluster redshift (and centre) from a field when no published value exists.

    Takes the peak of a smoothed redshift histogram, refines it with a biweight within +/- dv,
    and (if positions are given) returns the median position of galaxies near that redshift.
    Use only as a starting point for membership. Returns (z_seed, ra_c, dec_c) or None."""
    z = np.asarray(z, float)
    ok = (z > zmin) & (z < zmax)
    if ok.sum() < 5:
        return None
    bins = np.arange(zmin, zmax, bin_width)
    h, edges = np.histogram(z[ok], bins=bins)
    k = np.argmax(gaussian_filter1d(h.astype(float), smooth / bin_width))
    zpk = 0.5 * (edges[k] + edges[k + 1])
    near = velocity_window(z[ok], zpk, dv)
    zs = biweight_location(z[ok][near] if near.sum() >= 3 else z[ok])
    if ra is None:
        return zs, None, None
    close = velocity_window(z, zs, dv)
    sel = close if close.sum() >= 3 else np.ones(z.size, bool)
    return zs, float(np.median(np.asarray(ra)[sel])), float(np.median(np.asarray(dec)[sel]))


def aperture_members(z, R, mode='r200', r_fixed=1.5, r_start=1.5, tol=0.01, max_iter=50,
                     max_step=1.5, cosmo=PLANCK18):
    """Spatial + velocity membership.

    z, R : redshifts and projected radii (Mpc) of galaxies already inside the velocity window.
    mode : 'fixed' keeps R < r_fixed then clips; 'r200' iterates from r_start until R200
           converges to within tol. Each step changes the radius by at most a factor
           max_step, so a few galaxies entering at the edge cannot make it run away. If the
           radius flips between values (a limit cycle), the smallest radius in the cycle is used.

    Returns (mask into z, info) with info = dict(radius, n_iter, converged)."""
    z, R = np.asarray(z, float), np.asarray(R, float)

    def clip_inside(r):
        inside = np.where(R < r)[0]
        m = np.zeros(z.size, bool)
        if inside.size:
            m[inside[sigma_clip_members(z[inside])]] = True
        return m

    if mode == 'fixed':
        return clip_inside(r_fixed), dict(radius=r_fixed, n_iter=1, converged='fixed')
    if mode != 'r200':
        raise ValueError("mode must be 'fixed' or 'r200'")
    r, seen = r_start, []
    for it in range(1, max_iter + 1):
        if (R < r).sum() < 3:
            return R < r, dict(radius=r, n_iter=it, converged='too_few')
        mem = clip_inside(r)
        d = dynamics(z[mem])
        r_new = r200_from_m200(m200_from_sigma(d['sigma_v'], d['z'], cosmo), d['z'], cosmo)
        r_new = min(max(r_new, r / max_step), r * max_step)
        if abs(r_new - r) < tol:
            return mem, dict(radius=r_new, n_iter=it, converged='yes')
        if any(abs(r_new - s) < tol for s in seen):
            k = min(i for i, s in enumerate(seen) if abs(r_new - s) < tol)
            r_min = min(seen[k:] + [r])
            return clip_inside(r_min), dict(radius=r_min, n_iter=it, converged='cycle_min')
        seen.append(r)
        r = r_new
    return mem, dict(radius=r, n_iter=max_iter, converged='no')


def select_members(z, ra, dec, z_seed, ra0, dec0, mode='velocity', dv_max=5000.0,
                   cosmo=PLANCK18, **aperture_kw):
    """Full membership selection for one cluster field.

    mode: 'velocity', 'fixed' or 'r200'. Radii are computed at the redshift of the
    velocity-only members. Returns (mask into the input arrays, info dict)."""
    z, ra, dec = (np.asarray(a, float) for a in (z, ra, dec))
    win = np.where(velocity_window(z, z_seed, dv_max))[0]
    info = dict(n_window=win.size)
    mask = np.zeros(z.size, bool)
    if win.size < 3:
        return mask, dict(info, converged='too_few')
    vel = win[sigma_clip_members(z[win])]
    if mode == 'velocity':
        mask[vel] = True
        return mask, dict(info, radius=np.nan, converged='velocity')
    z_ref = biweight_location(z[vel])
    R = projected_distance(ra[win], dec[win], ra0, dec0, z_ref, cosmo)
    m, ainfo = aperture_members(z[win], R, mode=mode, cosmo=cosmo, **aperture_kw)
    mask[win[m]] = True
    return mask, dict(info, **ainfo)
