#!/usr/bin/env python
"""Pack many 1D spectra into ONE Marz file (Marz loads a single file at a time).

    python scripts/pack_marz.py out.fits spectra/*.fits [--clean] [--grid 4500,2,2652]
    python scripts/pack_marz.py blind.fits old/*.fits new/*.fits --blind key.csv

Each input is a 1D FITS spectrum with a linear wavelength solution in the header and,
ideally, an 'ERR' extension (1 sigma). All spectra are rebinned onto one grid (default: the
coarsest input sampling over the union of their ranges). Spectra without errors get a
variance from the scatter about a running median, so Marz's automatic fit still runs.

--clean       mask telluric bands, strong sky lines and outliers (see clusterspec/clean.py)
--windows F   text file of 'lo hi label' windows to mask instead of the defaults
--blind KEY   shuffle the spectra and name them blind_0001...; the mapping to the input
              files goes to KEY (keep it closed until you've measured everything)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from clusterspec.fitsio import read_fits                              # noqa: E402
from clusterspec.extract import wavelength_axis                        # noqa: E402
from clusterspec.marz import to_grid, write_marz                       # noqa: E402
from clusterspec.clean import clean_spectrum, DEFAULT_WINDOWS, read_windows   # noqa: E402


def load(path):
    H = read_fits(path)
    h, f = H[0]
    f = np.asarray(f, float).ravel() if f.ndim == 1 else np.asarray(f, float)[0]
    e = None
    for hh, d in H[1:]:
        if str(hh.get('EXTNAME', '')).upper() in ('ERR', 'ERROR', 'SIGMA') and d is not None:
            e = np.asarray(d, float).ravel()
    if e is None:                                        # no error spectrum: estimate one
        r = f - median_filter(np.nan_to_num(f), 21, mode='nearest')
        mad = median_filter(np.abs(np.nan_to_num(r)), 101, mode='nearest')
        e = 1.4826 * np.where(mad > 0, mad, np.nan)
    return wavelength_axis(h, f.size), f, e, h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('output')
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--grid', help='wave0,dwave,npix')
    ap.add_argument('--clean', action='store_true')
    ap.add_argument('--windows')
    ap.add_argument('--blind', metavar='KEY_CSV')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    spectra = [load(p) for p in a.inputs]
    if a.grid:
        w0, dw, n = a.grid.split(',')
        grid = float(w0) + float(dw) * np.arange(int(n))
    else:
        dw = max(np.diff(s[0]).mean() for s in spectra)
        lo, hi = min(s[0][0] for s in spectra), max(s[0][-1] for s in spectra)
        grid = lo + dw * np.arange(int((hi - lo) / dw) + 1)
    windows = read_windows(a.windows) if a.windows else DEFAULT_WINDOWS
    F = np.full((len(spectra), grid.size), np.nan)
    V = np.full_like(F, np.nan)
    for i, (w, f, e, h) in enumerate(spectra):
        if a.clean:
            f, e, _ = clean_spectrum(w, f, e, windows=windows)
        same = w.size == grid.size and np.allclose(w, grid)
        F[i], V[i] = (f, e ** 2) if same else to_grid(w, f, e ** 2, grid)
    names = [os.path.splitext(os.path.basename(p))[0] for p in a.inputs]
    order = np.arange(len(names))
    if a.blind:
        order = np.random.default_rng(a.seed).permutation(len(names))
        blind = [f'blind_{k + 1:04d}' for k in range(len(names))]
        pd.DataFrame(dict(name=blind, file=[a.inputs[i] for i in order])).to_csv(a.blind, index=False)
        names = blind
    else:
        names = [names[i] for i in order]
    write_marz(a.output, grid[0], grid[1] - grid[0], F[order], V[order], names)
    print(f'{len(names)} spectra -> {a.output} ({grid[0]:.1f}-{grid[-1]:.1f} A, '
          f'{grid[1] - grid[0]:.3g} A/pix, {np.isnan(F).mean():.1%} masked)')


if __name__ == '__main__':
    main()
