"""Minimal FLRW cosmology: H(z), comoving and angular-diameter distances.

Pass any object with ``H(z)`` and ``angular_diameter_distance(z)`` methods (in km/s/Mpc and
Mpc) wherever a ``cosmo`` argument is accepted, e.g. an astropy cosmology wrapped with
``AstropyCosmology`` below, if you prefer astropy.
"""
import numpy as np
from scipy.integrate import quad

C_KMS = 299792.458


class Cosmology:
    """Flat or curved LCDM with optional radiation density.

    H0 in km/s/Mpc; Om0, Ode0, Or0 are density parameters today. If Ode0 is None the model
    is flat (Ode0 = 1 - Om0 - Or0)."""

    def __init__(self, H0=70.0, Om0=0.3, Ode0=None, Or0=0.0):
        self.H0, self.Om0, self.Or0 = float(H0), float(Om0), float(Or0)
        self.Ode0 = 1.0 - Om0 - Or0 if Ode0 is None else float(Ode0)
        self.Ok0 = 1.0 - self.Om0 - self.Ode0 - self.Or0

    def E(self, z):
        z = np.asarray(z, float)
        return np.sqrt(self.Om0 * (1 + z) ** 3 + self.Or0 * (1 + z) ** 4
                       + self.Ok0 * (1 + z) ** 2 + self.Ode0)

    def H(self, z):
        return self.H0 * self.E(z)

    def comoving_distance(self, z):
        return C_KMS / self.H0 * quad(lambda x: 1.0 / self.E(x), 0, float(z))[0]

    def transverse_comoving_distance(self, z):
        dc = self.comoving_distance(z)
        dh = C_KMS / self.H0
        if abs(self.Ok0) < 1e-12:
            return dc
        sk = np.sqrt(abs(self.Ok0))
        return dh / sk * (np.sinh(sk * dc / dh) if self.Ok0 > 0 else np.sin(sk * dc / dh))

    def angular_diameter_distance(self, z):
        return self.transverse_comoving_distance(z) / (1 + float(z))

    def __repr__(self):
        return (f'Cosmology(H0={self.H0}, Om0={self.Om0}, Ode0={self.Ode0:.6f}, '
                f'Or0={self.Or0:.3g})')


class AstropyCosmology:
    """Wrap an astropy cosmology so it can be passed as ``cosmo``."""

    def __init__(self, c):
        self.c = c
        self.H0 = c.H0.value

    def H(self, z):
        return self.c.H(z).value

    def angular_diameter_distance(self, z):
        return self.c.angular_diameter_distance(z).value


# Planck 2018 H0 and Om0 (as in astropy.cosmology.Planck18), flat, radiation neglected
# (a < 0.01% effect on distances and H(z) at z < 1).
PLANCK18 = Cosmology(H0=67.66, Om0=0.30966)
