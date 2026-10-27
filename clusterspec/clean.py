"""Masking bad regions of 1D spectra before redshift fitting.

Automatic template fitters (Marz, autoz, ...) happily fit atmospheric absorption bands or
sky-subtraction residuals as if they were galaxy features. A typical failure: the telluric
A (7600 A) and B (6870 A) bands mimic H-alpha and Na D absorption at z ~ 0.16, so faint
spectra pile up at that redshift. Masked pixels are set to NaN, which Marz interpolates over
and gives no weight.
"""
import numpy as np
from scipy.ndimage import median_filter, binary_dilation

# (lo, hi, label) in Angstrom, observed frame, air. Edit for your site and instrument.
DEFAULT_WINDOWS = [
    (4353.0, 4363.0, 'Hg 4358 (street lights)'),
    (5456.0, 5466.0, 'Hg 5461 (street lights)'),
    (5572.0, 5582.0, '[OI] 5577 sky'),
    (6295.0, 6305.0, '[OI] 6300 sky'),
    (6359.0, 6369.0, '[OI] 6364 sky'),
    (6860.0, 6930.0, 'telluric B band'),
    (7170.0, 7350.0, 'telluric H2O'),
    (7590.0, 7700.0, 'telluric A band'),
]


def read_windows(path):
    """Read a 'lo hi [label]' text file of wavelength windows."""
    out = []
    for line in open(path):
        t = line.split()
        if len(t) >= 2 and not line.lstrip().startswith('#'):
            out.append((float(t[0]), float(t[1]), ' '.join(t[2:])))
    return out


def window_mask(wave, windows=DEFAULT_WINDOWS):
    """True inside any (lo, hi[, label]) window."""
    wave = np.asarray(wave, float)
    m = np.zeros(wave.size, bool)
    for w in windows:
        m |= (wave >= w[0]) & (wave <= w[1])
    return m


def sky_line_columns(err_spectra, threshold=3.0, grow=1, valid=None):
    """Columns dominated by strong sky lines, from a stack of error spectra on a common grid
    (e.g. all 1D errors of one mask, or the median error of the 2D frames).
    True where the median error exceeds ``threshold`` x its overall median, grown by ``grow``
    pixels. threshold=3 removes only the strongest lines (~1-2% of pixels in the red)."""
    e = np.nanmedian(np.atleast_2d(np.asarray(err_spectra, float)), axis=0)
    ref = np.nanmedian(e[valid] if valid is not None else e)
    m = np.nan_to_num(e / ref) > threshold
    return binary_dilation(m, iterations=grow) if grow else m


def zero_runs(flux, n=3):
    """True in runs of >= n exact zeros (typical of chip gaps filled with 0)."""
    z = np.asarray(flux) == 0
    m = np.zeros(z.size, bool)
    i = 0
    while i < z.size:
        if z[i]:
            j = i
            while j < z.size and z[j]:
                j += 1
            if j - i >= n:
                m[i:j] = True
            i = j
        else:
            i += 1
    return m


def outlier_mask(flux, err, width=41, nsig=5.0, spike_neighbour=2.0):
    """Impossible features relative to a running median of ``width`` pixels:
    dips below -nsig sigma (and their neighbours), and single-pixel spikes above +nsig sigma
    whose neighbours are both below spike_neighbour sigma. Real emission lines are resolved
    (several pixels), so they are not touched."""
    flux, err = np.asarray(flux, float), np.asarray(err, float)
    sm = median_filter(np.nan_to_num(flux), size=width, mode='nearest')
    with np.errstate(all='ignore'):
        r = np.where(np.isfinite(flux) & (err > 0), (flux - sm) / err, 0.0)
    dip = r < -nsig
    dip = dip | np.roll(dip, 1) | np.roll(dip, -1)
    spike = (r > nsig) & (np.roll(r, 1) < spike_neighbour) & (np.roll(r, -1) < spike_neighbour)
    return dip | spike


def clean_spectrum(wave, flux, err, windows=DEFAULT_WINDOWS, extra_mask=None, outliers=True,
                   **outlier_kw):
    """Apply window, extra (e.g. sky-line or chip-gap) and outlier masks.
    Returns (flux, err, mask) with masked pixels set to NaN."""
    f, e = np.array(flux, float), np.array(err, float)
    m = window_mask(wave, windows) if windows else np.zeros(f.size, bool)
    if extra_mask is not None:
        m |= extra_mask
    m |= ~np.isfinite(f) | ~np.isfinite(e) | (e <= 0)
    if outliers:
        m |= outlier_mask(np.where(m, np.nan, f), e, **outlier_kw)
    f[m] = np.nan
    e[m] = np.nan
    return f, e, m
