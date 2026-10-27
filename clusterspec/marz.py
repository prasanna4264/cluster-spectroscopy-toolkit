"""Marz (Hinton et al. 2016) input and output.

Marz loads one FITS file at a time, so to measure many spectra in one session they have to be
packed into a single multi-spectrum file:

    primary (EXTNAME INTENSITY)  flux, one row per spectrum, common wavelength grid
    VARIANCE                     same shape; without it Marz does not run its automatic fit
    FIBRES                       table: NAME, TYPE ('P' = keep), RA, DEC (radians)
    wavelength                   CRVAL1 / CDELT1 / CRPIX1 in the primary header

Marz assumes air wavelengths and converts to vacuum unless the header has VACUUM = T.
NaN pixels are interpolated over and given no weight.
"""
import io
import numpy as np
import pandas as pd
from .fitsio import write_fits


def rebin_matrix(wave_in, wave_out):
    """Sparse overlap matrix W (n_out, n_in) for flux-conserving rebinning between two grids of
    pixel centres: f_out = W @ f_in, var_out = W.power(2) @ var_in (fluxes per pixel)."""
    from scipy.sparse import coo_matrix

    def edges(w):
        d = np.diff(w)
        return np.r_[w[0] - d[0] / 2, w[:-1] + d / 2, w[-1] + d[-1] / 2]
    ei, eo = edges(np.asarray(wave_in, float)), edges(np.asarray(wave_out, float))
    n_in, n_out = len(ei) - 1, len(eo) - 1
    first = np.clip(np.searchsorted(ei, eo[:-1], 'right') - 1, 0, n_in - 1)
    last = np.clip(np.searchsorted(ei, eo[1:], 'left') - 1, 0, n_in - 1)
    rows, cols, vals = [], [], []
    for k in range(int((last - first).max()) + 1):
        i = first + k
        ok = i <= last
        j = np.where(ok)[0]
        i = i[ok]
        ov = np.minimum(eo[j + 1], ei[i + 1]) - np.maximum(eo[j], ei[i])
        keep = ov > 0
        rows.append(j[keep]); cols.append(i[keep])
        vals.append(ov[keep] / (ei[i + 1] - ei[i])[keep])
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n_out, n_in)).tocsr()


def to_grid(wave, flux, var, wave_out, min_cover=0.99):
    """Rebin one spectrum (flux per pixel) onto the pixel centres ``wave_out``.
    Output pixels that touch a NaN input pixel, or are less than ``min_cover`` covered by the
    input, are NaN. Note: rebinning correlates neighbouring pixels; recalibrate the variance
    afterwards (``clusterspec.extract.calibrate_errors``) if the absolute level matters."""
    wave, wave_out = np.asarray(wave, float), np.asarray(wave_out, float)
    W = rebin_matrix(wave, wave_out)
    f, v = np.asarray(flux, float), np.asarray(var, float)
    bad = ~np.isfinite(f) | ~np.isfinite(v)
    fo = W @ np.where(bad, 0, f)
    vo = W.power(2) @ np.where(bad, 0, v)
    ratio = np.mean(np.diff(wave_out)) / np.mean(np.diff(wave))
    cov = np.asarray(W.sum(1)).ravel() / ratio
    touched = (W @ bad.astype(float)) > 0
    out_bad = touched | (cov < min_cover)
    fo[out_bad] = np.nan
    vo[out_bad] = np.nan
    return fo, vo


def write_marz(path, wave0, dwave, flux, var, names, ra=None, dec=None, vacuum=False):
    """Write a multi-spectrum Marz file.

    flux, var : (n_spec, n_pix) arrays on the grid wave0 + dwave * arange(n_pix).
    names     : n_spec strings (Marz shows these and writes them to its results file).
    ra, dec   : optional, degrees (stored in radians, as Marz expects)."""
    flux = np.asarray(flux, float)
    var = np.asarray(var, float)
    n = flux.shape[0]
    bad = ~np.isfinite(flux) | ~np.isfinite(var) | (var <= 0)
    flux = np.where(bad, np.nan, flux)
    var = np.where(bad, np.nan, var)
    wcs = dict(CRVAL1=float(wave0), CDELT1=float(dwave), CD1_1=float(dwave), CRPIX1=1.0,
               CTYPE1='LINEAR')
    if vacuum:
        wcs['VACUUM'] = True
    width = max(16, max(len(str(s)) for s in names))
    rec = np.zeros(n, dtype=[('NAME', f'S{width}'), ('TYPE', 'S1'), ('RA', '>f8'), ('DEC', '>f8')])
    rec['NAME'] = [str(s).encode() for s in names]
    rec['TYPE'] = b'P'
    if ra is not None:
        rec['RA'] = np.radians(ra)
        rec['DEC'] = np.radians(dec)
    write_fits(path, [('image', flux, wcs, 'INTENSITY'), ('image', var, wcs, 'VARIANCE'),
                      ('table', rec, 'FIBRES')])


def read_marz_results(path):
    """Read a Marz results file (.mz or exported csv) into a DataFrame with columns
    name, z (final redshift), qop, template, xcor and the auto values."""
    lines = open(path).read().splitlines()
    hi = next(i for i, l in enumerate(lines) if 'QOP' in l.upper())
    txt = '\n'.join([lines[hi].lstrip('#').strip()]
                    + [l for l in lines[hi + 1:] if l.strip() and not l.startswith('#')])
    df = pd.read_csv(io.StringIO(txt), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}

    def col(*names):
        return next((low[n.lower()] for n in names if n.lower() in low), None)

    out = pd.DataFrame({
        'name': df[col('Name', 'ID')].astype(str).str.strip(),
        'z': pd.to_numeric(df[col('FinZ', 'Final z', 'z')], errors='coerce'),
        'qop': pd.to_numeric(df[col('QOP')], errors='coerce').fillna(0).astype(int)})
    for src, dst in (('FinTN', 'template'), ('AutoZ', 'z_auto'), ('AutoXCor', 'xcor'),
                     ('Comment', 'comment')):
        c = col(src)
        if c is not None:
            out[dst] = df[c].astype(str).str.strip() if dst in ('template', 'comment') \
                else pd.to_numeric(df[c], errors='coerce')
    return out


def telluric_pileup(z, qop, z_range=(0.14, 0.20), min_qop=3):
    """Number of fits with QOP >= min_qop inside z_range: a quick check for spectra fitted
    on the telluric A/B bands (see clean.py)."""
    z, qop = np.asarray(z, float), np.asarray(qop)
    return int(((z >= z_range[0]) & (z <= z_range[1]) & (qop >= min_qop)).sum())
