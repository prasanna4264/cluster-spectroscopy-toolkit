import numpy as np
from clusterspec.extract import (extract_slit, extract_mask, trace_slit, bad_pixel_mask,
                                 aperture_extract)

NY, NX = 34, 1500


def slit(center=15.0, tilt=0.8, sigma=2.2, flux=300.0, noise=4.0, seed=0, gap=True):
    rng = np.random.default_rng(seed)
    y = np.arange(NY)[:, None]
    tr = center + tilt * (np.arange(NX) - NX / 2) / NX
    p = np.exp(-0.5 * ((y - tr[None]) / sigma) ** 2)
    p /= p.sum(0, keepdims=True)
    err = np.full((NY, NX), noise)
    f = p * flux + rng.normal(0, noise, (NY, NX))
    if gap:
        f[:, 700:720] = 0.0
        err[:, 700:720] = -9999.0
    return f, err, tr


def test_masks_find_gap():
    f, e, _ = slit()
    bad, gap = bad_pixel_mask(f, e)
    assert gap[700:720].all() and not gap[:700].any()


def test_trace_recovered():
    f, e, tr = slit()
    bad, _ = bad_pixel_mask(f, e)
    t, meth = trace_slit(f, 15.0, 2.2, bad)
    assert meth == 'poly2' and np.max(np.abs(t - tr)) < 0.5


def test_flux_and_errors_both_methods():
    for method, tol in (('optimal', 0.03), ('aperture', 0.15)):
        fl, z = [], []
        for s in range(30):
            f, e, _ = slit(seed=s)
            r = extract_slit(f, e, method=method)
            ok = np.isfinite(r['flux'])
            fl.append(np.median(r['flux'][ok]))
            z.append(np.std((r['flux'][ok] - 300.0) / r['err'][ok]))
        frac = 1.0 if method == 'optimal' else 0.866       # +/-1.5 sigma box holds 86.6%
        assert abs(np.median(fl) / (300 * frac) - 1) < tol, method
        assert abs(np.median(z) - 1) < 0.1, method
        assert np.isnan(r['flux'][700:720]).all()


def test_wrong_trace_loses_flux():
    f, e, tr = slit(center=12.0)
    good = extract_slit(f, e, method='aperture')['flux']
    shifted = aperture_extract(f, tr + 4.0, 1.5 * 2.2)[0]
    assert np.nanmedian(shifted) < 0.5 * np.nanmedian(good)        # 4 px off: under half the flux


def test_mask_fallback_for_faint_slit():
    slits = [dict(flux2d=slit(center=16.0, seed=s)[0], err2d=slit(seed=s)[1]) for s in range(4)]
    faint = slit(center=16.0, flux=5.0, seed=9)
    slits.append(dict(flux2d=faint[0], err2d=faint[1]))
    res = extract_mask(slits)
    assert abs(res[-1]['center'] - 16.0) < 1.5
