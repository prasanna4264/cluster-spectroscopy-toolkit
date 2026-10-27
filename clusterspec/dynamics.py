"""Cluster redshift, velocity dispersion, dynamical mass and phase-space coordinates."""
import numpy as np
from .stats import biweight_location, biweight_scale, jackknife
from .cosmo import PLANCK18, C_KMS

G_MPC = 4.30091e-9          # gravitational constant in Mpc (km/s)^2 / Msun


def peculiar_velocity(z, z_cl):
    """Rest-frame line-of-sight velocity relative to the cluster (km/s)."""
    return C_KMS * (np.asarray(z, float) - z_cl) / (1 + z_cl)


def angular_separation(ra, dec, ra0, dec0):
    """Angular separation in radians (Vincenty formula; all inputs in degrees)."""
    ra, dec = np.radians(ra), np.radians(dec)
    ra0, dec0 = np.radians(ra0), np.radians(dec0)
    dra = ra - ra0
    num = np.hypot(np.cos(dec) * np.sin(dra),
                   np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(dra))
    den = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(dra)
    return np.arctan2(num, den)


def projected_distance(ra, dec, ra0, dec0, z, cosmo=PLANCK18):
    """Projected physical distance (Mpc) from (ra0, dec0) at redshift z."""
    return angular_separation(ra, dec, ra0, dec0) * cosmo.angular_diameter_distance(z)


def dynamics(z):
    """Biweight redshift and velocity dispersion of a member sample, with jackknife errors.

    Returns a dict: z (biweight location), z_err, z_ci (95%, centred on z), z_scale,
    sigma_v and sigma_v_err (km/s, rest frame), N."""
    z = np.asarray(z, float)
    zb, sb = biweight_location(z), biweight_scale(z)
    _, _, z_err = jackknife(z, biweight_location)
    _, _, s_err = jackknife(z, biweight_scale)
    return dict(z=zb, z_err=z_err, z_ci=(zb - 1.96 * z_err, zb + 1.96 * z_err), z_scale=sb,
                sigma_v=C_KMS * sb / (1 + zb), sigma_v_err=C_KMS * s_err / (1 + zb), N=z.size)


def m200_from_sigma(sigma_v, z, cosmo=PLANCK18, norm=1082.9, alpha=0.3361):
    """M200 (Msun) from the velocity dispersion with the Evrard et al. (2008) relation
    sigma_DM = norm * (h(z) M200 / 1e15 Msun)^alpha. Note this is the dark-matter relation;
    galaxy velocity bias is not included."""
    hz = cosmo.H(z) / cosmo.H0
    return (sigma_v / norm) ** (1 / alpha) * 1e15 / hz


def r200_from_m200(m200, z, cosmo=PLANCK18):
    """R200 (Mpc): radius enclosing 200x the critical density at z."""
    rho_c = 3 * cosmo.H(z) ** 2 / (8 * np.pi * G_MPC)
    return (3 * m200 / (4 * np.pi * 200 * rho_c)) ** (1 / 3)
