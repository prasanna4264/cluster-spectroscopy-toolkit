"""clusterspec: multi-slit spectral extraction and galaxy-cluster kinematics.

Modules
-------
extract   1D extraction from 2D multi-slit spectra (aperture and Horne optimal)
clean     masking telluric bands, sky lines, chip gaps and outliers in 1D spectra
marz      packing spectra into one Marz file; reading Marz results
fitsio    small dependency-free FITS reader/writer
members   cluster membership (velocity window, biweight clipping, fixed / R200 apertures)
dynamics  biweight redshift and dispersion, M200 (Evrard+08), R200, phase-space coordinates
ds        Dressler-Shectman substructure test with Monte Carlo calibration
power     DS detection power from mock velocities on real positions
pipeline  analyse_cluster / run_catalog
plots     phase-space, DS histogram, DS bubble plots
"""
__version__ = '0.1.0'

from .cosmo import Cosmology, PLANCK18
from .pipeline import analyse_cluster, run_catalog
