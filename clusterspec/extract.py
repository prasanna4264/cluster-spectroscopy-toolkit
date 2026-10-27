"""1D extraction from rectified, sky-subtracted 2D multi-slit spectra.

Steps for one slitlet (each is a function you can swap out):

1. bad_pixel_mask     flagged / non-finite pixels and fully dead columns (chip gaps)
2. spatial_profile    median-collapse along wavelength, Gaussian + linear fit -> centre, width
3. trace_slit         flux-weighted centroids in wavelength bins + low-order polynomial
4. error_scale        compare the quoted error image with the real scatter in sky rows
5. extraction         aperture_extract (boxcar with fractional pixels) or
                      optimal_extract (Horne 1986, empirical profile, variance weights)
6. calibrate_errors   rescale the 1D error so residual scatter = 1 sigma
                      (neighbouring pixels in rectified data are correlated)

``extract_mask`` runs these over every slitlet of a mask, using mask-wide information for
faint slits (median object row, median error scale). Defaults are what worked for
Magellan/IMACS data reduced with COSMOS; see docs/extraction.md for what to change.
"""
import warnings
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.ndimage import median_filter

BAD_VALUE = -9999.0


# 1. masks -------------------------------------------------------------------------------
def bad_pixel_mask(flux2d, err2d=None, bad_value=BAD_VALUE):
    """True where a pixel should not be used. Columns where every pixel is zero or bad
    (chip gaps, unexposed ends) are masked entirely. Returns (bad, gap_columns)."""
    f = np.asarray(flux2d, float)
    bad = ~np.isfinite(f) | (f == bad_value)
    if err2d is not None:
        e = np.asarray(err2d, float)
        bad |= ~np.isfinite(e) | (e <= 0) | (e == bad_value)
    gap = np.all((f == 0) | bad, axis=0)
    return bad | gap[None, :], gap


# 2. spatial profile ---------------------------------------------------------------------
def _gauss_lin(y, a, m, s, c0, c1):
    return a * np.exp(-0.5 * ((y - m) / s) ** 2) + c0 + c1 * y


def spatial_profile(flux2d, bad=None, cols=(0.15, 0.85), edge=2, sigma_guess=2.5,
                    sigma_range=(0.6, 8.0), fallback_center=None):
    """Fit Gaussian + linear background to the median spatial profile.

    cols  : fraction of the wavelength axis to collapse (avoid noisy ends).
    edge  : rows at each slit end ignored when looking for the peak (edge artefacts).
    If the fit fails or is implausible, the centre falls back to ``fallback_center``
    (e.g. the median object row on the mask) or the profile peak.
    Returns dict(center, sigma, fit_ok, profile)."""
    ny, nx = flux2d.shape
    d = np.where(bad, np.nan, flux2d) if bad is not None else np.asarray(flux2d, float)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        prof = np.nan_to_num(np.nanmedian(d[:, int(cols[0] * nx):int(cols[1] * nx)], axis=1))
    y = np.arange(ny)
    pk = edge + int(np.argmax(prof[edge:ny - edge]))
    p0 = [prof[pk] - np.median(prof), float(pk), sigma_guess, np.median(prof), 0.0]
    ok = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', OptimizeWarning)
            p, _ = curve_fit(_gauss_lin, y, prof, p0=p0, maxfev=20000)
        ok = (p[0] > 0 and edge <= p[1] <= ny - 1 - edge
              and sigma_range[0] < abs(p[2]) < sigma_range[1])
    except Exception:
        pass
    if ok:
        return dict(center=float(p[1]), sigma=float(abs(p[2])), fit_ok=True, profile=prof)
    c = float(fallback_center) if fallback_center is not None else float(pk)
    return dict(center=c, sigma=float(sigma_guess), fit_ok=False, profile=prof)


# 3. trace -------------------------------------------------------------------------------
def trace_slit(flux2d, center, sigma, bad=None, nbins=30, deg=2, min_snr=4.0, halfwidth=3.0,
               max_offset=None):
    """Object position along the wavelength axis.

    Flux-weighted centroid within +/- halfwidth*sigma of ``center`` in ``nbins`` wavelength
    bins; bins with peak S/N < min_snr are skipped; polynomial of degree ``deg`` with one
    3-sigma clip. Faint objects (too few bins) or wild fits (further than max_offset pixels
    from ``center``; default 4*sigma + 2) get a flat trace at ``center``.
    Returns (trace[nx], method string)."""
    ny, nx = flux2d.shape
    y = np.arange(ny)
    d = np.where(bad, np.nan, flux2d) if bad is not None else np.asarray(flux2d, float)
    lo = max(0, int(center - halfwidth * sigma))
    hi = min(ny, int(np.ceil(center + halfwidth * sigma)) + 1)
    edges = np.linspace(0, nx, nbins + 1).astype(int)
    xs, ys = [], []
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for a, b in zip(edges[:-1], edges[1:]):
            col = np.nanmedian(d[:, a:b], axis=1)
            if np.all(np.isnan(col[lo:hi])):
                continue
            bg = np.nanmedian(d[:, a:b])
            pr = np.nan_to_num(col[lo:hi] - bg)
            w = np.clip(pr, 0, None)
            noise = 1.4826 * np.nanmedian(np.abs(col - bg)) + 1e-9
            if w.sum() > 0 and pr.max() / noise > min_snr:
                xs.append(0.5 * (a + b))
                ys.append(np.sum(y[lo:hi] * w) / w.sum())
    xs, ys = np.array(xs), np.array(ys)
    if len(xs) < deg + 4:
        return np.full(nx, float(center)), 'flat'
    c = np.polyfit(xs, ys, deg)
    r = ys - np.polyval(c, xs)
    k = np.abs(r) < 3 * (np.std(r) + 0.05)
    tr = np.polyval(np.polyfit(xs[k], ys[k], deg), np.arange(nx))
    max_offset = 4 * sigma + 2 if max_offset is None else max_offset
    if np.any(np.abs(tr - center) > max_offset):
        return np.full(nx, float(center)), 'flat (fit rejected)'
    return tr, f'poly{deg}'


# 4. error image check --------------------------------------------------------------------
def error_scale(flux2d, err2d, bad, center, sigma, width=31):
    """Ratio of the real pixel scatter in object-free rows to the quoted error image.
    > 1 means the error image underestimates the noise. NaN if too few sky pixels."""
    ny = flux2d.shape[0]
    rows = [r for r in range(1, ny - 1) if abs(r - center) > 3 * sigma + 2]
    if len(rows) < 3:
        return np.nan
    d = np.where(bad, np.nan, flux2d)[rows]
    e = np.where(bad, np.nan, err2d)[rows]
    z = (d - median_filter(np.nan_to_num(d), size=(1, width))) / e
    z = z[np.isfinite(z)]
    return float(1.4826 * np.median(np.abs(z - np.median(z)))) if z.size > 200 else np.nan


# 5a. aperture ----------------------------------------------------------------------------
def aperture_extract(flux2d, trace, halfwidth, err2d=None, bad=None, min_cover=0.6):
    """Sum within +/- halfwidth pixels of the trace, with fractional edge pixels.
    Masked pixels are left out and the sum is scaled up by the covered fraction; columns
    with less than ``min_cover`` of the aperture usable are NaN. Returns (flux, err)."""
    ny, nx = flux2d.shape
    y = np.arange(ny)[:, None]
    w = np.clip(np.minimum(y + 0.5, trace[None] + halfwidth)
                - np.maximum(y - 0.5, trace[None] - halfwidth), 0, 1)
    bad = np.zeros(flux2d.shape, bool) if bad is None else bad
    wb = np.where(bad, 0.0, w)
    cover = wb.sum(0) / np.clip(w.sum(0), 1e-9, None)
    with np.errstate(all='ignore'):
        f = np.nansum(wb * np.where(bad, 0, flux2d), 0) / cover
        e = (np.sqrt(np.nansum((wb * np.where(bad, 0, err2d)) ** 2, 0)) / cover
             if err2d is not None else np.full(nx, np.nan))
    f[cover < min_cover] = np.nan
    e[cover < min_cover] = np.nan
    return f, e


# 5b. optimal -----------------------------------------------------------------------------
def optimal_extract(flux2d, err2d, trace, sigma, bad=None, window=3.0, profile='empirical',
                    smooth=101, cr_clip=None, cr_sharp=6.0, n_iter=5, min_cover=0.5):
    """Horne (1986) optimal extraction.

    profile  : 'empirical' (data / column sum, median-smoothed over ``smooth`` columns along
               wavelength, recommended) or 'gaussian' (width ``sigma`` around the trace).
    cr_clip  : None (default) = no cosmic-ray rejection. Use it only on data that were NOT
               cleaned in 2D; a pixel is rejected only if it is a > cr_clip outlier from the
               profile model AND a > cr_sharp outlier from its neighbours along wavelength,
               so real emission lines (several pixels wide) survive.
    Returns (flux, err, n_rejected)."""
    ny, nx = flux2d.shape
    y = np.arange(ny)[:, None]
    bad = np.zeros(flux2d.shape, bool) if bad is None else bad
    inwin = np.abs(y - trace[None]) <= window * sigma + 0.5
    good = inwin & ~bad
    D = np.where(good, flux2d, 0.0)
    S = np.where(good, err2d, np.inf)
    V = S ** 2
    if profile == 'empirical':
        fbox = D.sum(0)
        with np.errstate(all='ignore'):
            P = np.where(good, D / np.where(np.abs(fbox) > 0, fbox, np.nan)[None], np.nan)
        P = np.array([median_filter(np.nan_to_num(r), size=smooth, mode='nearest') for r in P])
        P = np.clip(P, 0, None) * inwin
    else:
        P = np.zeros(flux2d.shape)
    if P.sum() == 0:
        P = np.exp(-0.5 * ((y - trace[None]) / sigma) ** 2) * inwin
    P /= np.clip(P.sum(0, keepdims=True), 1e-12, None)
    if cr_clip is not None:
        Dn = np.where(bad, 0.0, flux2d)
        sharp = (Dn - median_filter(Dn, size=(1, 7), mode='nearest')) > cr_sharp * S
    n_rej = 0
    for _ in range(n_iter):
        num = np.sum(np.where(good, P * D / V, 0), 0)
        den = np.sum(np.where(good, P ** 2 / V, 0), 0)
        f = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        if cr_clip is None:
            break
        res = np.where(good, (D - P * np.nan_to_num(f)[None]) / S, 0)
        cand = good & (res > cr_clip) & sharp
        if not cand.any():
            break
        worst = np.argmax(np.where(cand, res, -np.inf), axis=0)
        cols = np.where(cand.any(0))[0]
        good[worst[cols], cols] = False
        n_rej += cols.size
    e = np.where(den > 0, 1 / np.sqrt(np.where(den > 0, den, 1)), np.nan)
    cover = np.sum(np.where(good, P, 0), 0)
    f[cover < min_cover] = np.nan
    e[cover < min_cover] = np.nan
    return f, e, n_rej


# 6. 1D error calibration -----------------------------------------------------------------
def calibrate_errors(flux, err, valid=None, width=41, limits=(0.5, 4.0)):
    """Factor by which ``err`` should be multiplied so that residuals about a running median
    have unit scatter. Returns NaN if it cannot be measured or falls outside ``limits``."""
    flux, err = np.asarray(flux, float), np.asarray(err, float)
    ok = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    if valid is not None:
        ok &= valid
    if ok.sum() < 300:
        return np.nan
    sm = median_filter(np.nan_to_num(flux), size=width, mode='nearest')
    z = ((flux - sm) / err)[ok]
    s = 1.4826 * np.median(np.abs(z - np.median(z)))
    return float(s) if limits[0] < s < limits[1] else np.nan


def wavelength_axis(header, nx):
    """Linear wavelength grid from CRVAL1, CDELT1 (or CD1_1) and CRPIX1."""
    d = float(header.get('CDELT1', header.get('CD1_1')))
    return float(header['CRVAL1']) + (np.arange(nx) + 1 - float(header.get('CRPIX1', 1))) * d


# whole slit / whole mask -----------------------------------------------------------------
def extract_slit(flux2d, err2d=None, method='optimal', aperture_k=1.5, fallback_center=None,
                 err_scale=None, trace=None, valid=None, profile_kw=None, trace_kw=None,
                 optimal_kw=None):
    """Steps 1-6 for one slitlet. Returns a dict with flux, err and diagnostics.

    method      : 'optimal' (needs err2d) or 'aperture'.
    err_scale   : multiply err2d by this (e.g. the mask median from ``error_scale``);
                  if None, measured on this slit.
    trace       : supply a trace instead of fitting one.
    valid       : boolean mask of wavelength columns to use when calibrating the 1D errors."""
    flux2d = np.asarray(flux2d, float)
    err2d = None if err2d is None else np.asarray(err2d, float)
    bad, gap = bad_pixel_mask(flux2d, err2d)
    prof = spatial_profile(flux2d, bad, fallback_center=fallback_center, **(profile_kw or {}))
    c, s = prof['center'], prof['sigma']
    if trace is None:
        trace, tmeth = trace_slit(flux2d, c, s, bad, **(trace_kw or {}))
    else:
        tmeth = 'supplied'
    if err2d is not None:
        sc = error_scale(flux2d, err2d, bad, c, s) if err_scale is None else err_scale
        e2 = err2d * (sc if np.isfinite(sc) else 1.0)
    else:
        sc, e2 = np.nan, None
    if method == 'optimal':
        if e2 is None:
            raise ValueError('optimal extraction needs an error image')
        f, e, nrej = optimal_extract(flux2d, e2, trace, s, bad, **(optimal_kw or {}))
    elif method == 'aperture':
        f, e = aperture_extract(flux2d, trace, aperture_k * s, e2, bad)
        nrej = 0
    else:
        raise ValueError("method must be 'optimal' or 'aperture'")
    cal = calibrate_errors(f, e, valid) if e2 is not None else np.nan
    if np.isfinite(cal):
        e = e * cal
    return dict(flux=f, err=e, trace=trace, center=c, sigma=s, profile_fit_ok=prof['fit_ok'],
                trace_method=tmeth, err_scale_2d=sc, err_scale_1d=cal, n_rejected=nrej,
                gap_columns=gap)


def extract_mask(slits, method='optimal', aperture_k=1.5, valid=None, **kw):
    """Extract every slitlet of a mask.

    slits : list of dicts with 'flux2d', optional 'err2d', and anything else you want carried
            through (e.g. 'header', 'object', 'slit').
    Uses two mask-wide numbers for robustness: the median object row (fallback centre for
    slits whose profile can't be fitted) and the median error-image scale (used when a slit
    has too few sky pixels). Returns a list of result dicts (input keys + extraction)."""
    first = []
    for s in slits:
        bad, _ = bad_pixel_mask(s['flux2d'], s.get('err2d'))
        p = spatial_profile(np.asarray(s['flux2d'], float), bad)
        sc = (error_scale(np.asarray(s['flux2d'], float), np.asarray(s['err2d'], float), bad,
                          p['center'], p['sigma']) if s.get('err2d') is not None else np.nan)
        first.append((p, sc))
    centers = [p['center'] for p, _ in first if p['fit_ok']]
    mask_center = float(np.median(centers)) if centers else None
    scales = np.array([sc for _, sc in first], float)
    med_scale = float(np.nanmedian(scales)) if np.isfinite(scales).any() else np.nan
    out = []
    for s, (p, sc) in zip(slits, first):
        use = sc if np.isfinite(sc) and 0.5 < sc < 3 else med_scale
        r = extract_slit(s['flux2d'], s.get('err2d'), method=method, aperture_k=aperture_k,
                         fallback_center=mask_center, err_scale=use, valid=valid, **kw)
        out.append({**{k: v for k, v in s.items() if k not in ('flux2d', 'err2d')}, **r})
    return out
