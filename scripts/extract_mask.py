#!/usr/bin/env python
"""Extract 1D spectra from a mask-level 2D file (one FITS extension per slitlet).

    python scripts/extract_mask.py mask_2spec.fits out_dir/ [--method optimal|aperture]

Writes <object>_slit<N>.fits (flux + ERR extension) for every slitlet, extraction_log.csv,
and, with --qa, a PNG per slit showing the 2D spectrum, trace and 1D result. The input file is
only read. See docs/extraction.md for the steps and what to check on new data.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from clusterspec.fitsio import read_fits, write_spectrum           # noqa: E402
from clusterspec.extract import extract_mask, wavelength_axis       # noqa: E402

KEEP = ('OBJECT', 'SLITNUM', 'SLITMASK', 'DATE-OBS', 'EXPTIME', 'TELESCOP', 'INSTRUME',
        'DISPERSR', 'FILTER', 'BUNIT')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('outdir')
    ap.add_argument('--method', default='optimal', choices=['optimal', 'aperture'])
    ap.add_argument('--aperture-k', type=float, default=1.5,
                    help='aperture half-width in units of the profile sigma (aperture method)')
    ap.add_argument('--flux-plane', type=int, default=0)
    ap.add_argument('--err-plane', type=int, default=1, help='-1 if there is no error plane')
    ap.add_argument('--wmin', type=float, default=None, help='ignore wavelengths below this '
                    'when calibrating errors (e.g. a low-throughput blue end)')
    ap.add_argument('--qa', action='store_true', help='write a QA plot per slit')
    a = ap.parse_args()

    slits = []
    for ext, (h, d) in enumerate(read_fits(a.input)):
        if d is None or d.ndim not in (2, 3):
            continue
        f2 = d[a.flux_plane] if d.ndim == 3 else d
        e2 = d[a.err_plane] if (d.ndim == 3 and a.err_plane >= 0) else None
        slits.append(dict(flux2d=f2, err2d=e2, header=h, ext=ext,
                          object=str(h.get('OBJECT', f'ext{ext}')).strip() or f'ext{ext}',
                          slit=h.get('SLITNUM', ext)))
    if not slits:
        sys.exit('no 2D or 3D image extensions found')
    if a.method == 'optimal' and any(s['err2d'] is None for s in slits):
        sys.exit('optimal extraction needs an error plane; use --method aperture')
    nx = slits[0]['flux2d'].shape[1]
    wave = wavelength_axis(slits[0]['header'], nx)
    valid = np.ones(nx, bool)
    valid[:100] = valid[-100:] = False
    if a.wmin:
        valid &= wave > a.wmin
    res = extract_mask(slits, method=a.method, aperture_k=a.aperture_k, valid=valid)

    os.makedirs(a.outdir, exist_ok=True)
    log = ['ext,object,slit,file,center,sigma,profile_fit_ok,trace_method,err_scale_2d,err_scale_1d']
    for s, r in zip(slits, res):
        h = s['header']
        w = wavelength_axis(h, s['flux2d'].shape[1])
        name = f"{r['object']}_slit{r['slit']}.fits"
        keys = {k: h[k] for k in KEEP if k in h}
        keys.update(EXTMETH=a.method, PROFCEN=round(r['center'], 2), PROFSIG=round(r['sigma'], 3),
                    TRACEMTH=r['trace_method'])
        write_spectrum(os.path.join(a.outdir, name), w[0], w[1] - w[0], r['flux'], r['err'], keys)
        log.append(f"{s['ext']},{r['object']},{r['slit']},{name},{r['center']:.2f},{r['sigma']:.3f},"
                   f"{r['profile_fit_ok']},{r['trace_method']},{r['err_scale_2d']:.3f},{r['err_scale_1d']:.3f}")
        if a.qa:
            qa_plot(os.path.join(a.outdir, name.replace('.fits', '.png')), s['flux2d'], w, r, name)
    open(os.path.join(a.outdir, 'extraction_log.csv'), 'w').write('\n'.join(log) + '\n')
    nflat = sum(r['trace_method'].startswith('flat') for r in res)
    print(f'{len(res)} slitlets extracted to {a.outdir} ({nflat} with a flat trace: faint or untraceable)')


def qa_plot(path, f2, w, r, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    lo, hi = np.nanpercentile(f2, [5, 99])
    ax[0].imshow(f2, aspect='auto', origin='lower', cmap='gray', vmin=lo, vmax=hi,
                 extent=[w[0], w[-1], -0.5, f2.shape[0] - 0.5])
    ax[0].plot(w, r['trace'], 'r', lw=0.7)
    ax[1].plot(w, r['flux'], 'k', lw=0.5)
    ax[1].plot(w, r['err'], 'C3', lw=0.5)
    ax[1].set_xlabel('wavelength (A)')
    ax[0].set_title(f"{title}   centre {r['center']:.1f}  sigma {r['sigma']:.2f}  {r['trace_method']}")
    fig.savefig(path, dpi=80)
    plt.close(fig)


if __name__ == '__main__':
    main()
