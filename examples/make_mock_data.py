#!/usr/bin/env python
"""Make small mock data sets so every script in the repo can be tried without real data.

    python examples/make_mock_data.py

Writes to examples/data/:
  mock_mask_2spec.fits   12 rectified, sky-subtracted 2D slitlets (flux + error planes, one
                         extension each) with emission-line galaxies, a chip gap, flagged
                         pixels and a few faint objects
  mock_galaxies.csv      redshift catalogue of three mock cluster fields (cluster, z, ra, dec)
  mock_centres.csv       "published" redshifts/centres for two of them (the third is seeded
                         from the data)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from clusterspec.fitsio import write_fits                             # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
C = 299792.458
rng = np.random.default_rng(42)


def mock_mask(path, n_slits=12, ny=34, nx=2400, w0=4500.0, dw=2.0):
    wave = w0 + dw * np.arange(nx)
    sky = 1 + 8 * np.exp(-0.5 * ((wave[:, None] - np.array([5577, 6300, 7341, 7913, 8344, 8827, 9375])) / 3) ** 2).sum(1)
    lines = [(3727.4, 6.0), (4861.3, 2.0), (4958.9, 1.5), (5006.8, 4.0), (6562.8, 5.0)]
    hdus = [('image', np.zeros(0), {'INSTRUME': 'MOCK'})]
    for k in range(n_slits):
        z = 0.45 + rng.normal(0, 0.003) if k % 4 else rng.uniform(0.2, 0.7)
        bright = 0.3 if k in (3, 9) else 1.0                          # two faint objects
        spec = bright * (20 + 4 * (wave > 4000 * (1 + z)))           # continuum with a 4000 A break
        for lam, amp in lines:
            spec = spec + bright * 25 * amp * np.exp(-0.5 * ((wave - lam * (1 + z)) / 3.0) ** 2)
        y = np.arange(ny)[:, None]
        cen = 12 + 10 * rng.random() + 0.4 * (wave - wave.mean()) / 2000      # slightly tilted
        prof = np.exp(-0.5 * ((y - cen[None]) / 2.2) ** 2)
        prof /= prof.sum(0, keepdims=True)
        err = np.sqrt(4.0 * sky[None, :] + 0.2 * prof * spec[None, :]) * np.ones((ny, 1))
        flux = prof * spec[None, :] + rng.normal(0, 1, (ny, nx)) * err
        flux[:, 1200:1225] = 0.0                                       # chip gap
        err[:, 1200:1225] = -9999.0
        bad = rng.random((ny, nx)) < 0.002                             # flagged pixels
        flux[bad] = 0.0
        err[bad] = -9999.0
        hdr = dict(CRVAL1=w0, CDELT1=dw, CRPIX1=1.0, OBJECT=f'gal{100 + k}', SLITNUM=k + 1,
                   ZTRUE=round(float(z), 5))
        hdus.append(('image', np.stack([flux, err]), hdr))
    write_fits(path, hdus)


def mock_field(name, z_cl, ra0, dec0, n_mem, sigma, sub=None, n_field=40, r_scale=0.03):
    """Cluster members (Gaussian velocities, projected King-like radii) plus foreground,
    background and lensed-arc redshifts. sub = (fraction, dv/sigma, offset_deg) adds a
    subgroup offset on the sky and in velocity."""
    r = r_scale * np.sqrt(rng.random(n_mem) / (1 - rng.random(n_mem) * 0.9))
    th = rng.uniform(0, 2 * np.pi, n_mem)
    ra = ra0 + r * np.cos(th) / np.cos(np.radians(dec0))
    dec = dec0 + r * np.sin(th)
    v = rng.normal(0, sigma, n_mem)
    if sub:
        f, dvs, off = sub
        k = int(f * n_mem)
        ra[:k] = ra0 + off + rng.normal(0, 0.01, k)
        dec[:k] = dec0 + rng.normal(0, 0.01, k)
        v[:k] = rng.normal(dvs * sigma, 0.4 * sigma, k)
    z = z_cl + v * (1 + z_cl) / C
    zf = np.r_[rng.uniform(0.05, 1.0, n_field), rng.uniform(1.5, 3.0, 3)]
    raf = ra0 + rng.uniform(-0.15, 0.15, zf.size)
    decf = dec0 + rng.uniform(-0.15, 0.15, zf.size)
    return pd.DataFrame(dict(cluster=name, z=np.r_[z, zf].round(5), ra=np.r_[ra, raf].round(6),
                             dec=np.r_[dec, decf].round(6)))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    mock_mask(os.path.join(OUT, 'mock_mask_2spec.fits'))
    cat = pd.concat([
        mock_field('MOCK-A relaxed', 0.35, 150.0, 2.0, 70, 900),
        mock_field('MOCK-B merging', 0.48, 210.0, -5.0, 80, 1000, sub=(0.3, 1.8, 0.03)),
        mock_field('MOCK-C no published z', 0.62, 330.0, 10.0, 30, 700),
    ], ignore_index=True)
    cat.to_csv(os.path.join(OUT, 'mock_galaxies.csv'), index=False)
    pd.DataFrame(dict(cluster=['MOCK-A relaxed', 'MOCK-B merging'], z=[0.35, 0.48],
                      ra=[150.0, 210.0], dec=[2.0, -5.0])).to_csv(
        os.path.join(OUT, 'mock_centres.csv'), index=False)
    print('mock data written to', OUT)
