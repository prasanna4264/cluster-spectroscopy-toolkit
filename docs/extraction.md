# Extraction and redshift preparation

The extraction assumes the 2D spectra are already **reduced, sky-subtracted and rectified**
(wavelength along x, linear wavelength solution in the header), with one FITS extension per
slitlet. For IMACS that is the `*_2spec.fits` output of COSMOS: each extension has shape
`(2, ny, nx)`, plane 0 the flux and plane 1 the 1-sigma error, with −9999 marking bad pixels.

Every step below is a separate function in `clusterspec/extract.py`, so any of them can be
replaced. The defaults are what worked for our data; the "change if" column is the part to
think about for yours.

| # | step | function | default | change if |
|---|---|---|---|---|
| 1 | bad pixels and dead columns | `bad_pixel_mask` | non-finite, −9999, error ≤ 0; columns that are all zero/bad (chip gaps) | your pipeline flags bad pixels differently (e.g. a separate mask extension, NaN) |
| 2 | spatial profile | `spatial_profile` | median over the middle 70% of the wavelength axis, Gaussian + linear background; ignore 2 rows at each slit end | objects are extended or there are two objects in a slit (fit two components, or cut the slit); slits are very short |
| 3 | trace | `trace_slit` | flux-weighted centroids in 30 wavelength bins, keep bins with peak S/N > 4, quadratic fit, one 3σ clip; flat trace if too few bins | traces are strongly curved (raise the degree) or the data are very faint (fewer, wider bins) |
| 4 | error image check | `error_scale` | robust scatter of object-free rows ÷ quoted error | no object-free rows (very short slits): the mask median is used |
| 5a | aperture extraction | `aperture_extract` | ±1.5σ of the profile, fractional edge pixels, masked pixels corrected for | you want a fixed width in pixels |
| 5b | optimal extraction | `optimal_extract` | Horne (1986), empirical profile smoothed over 101 columns, no cosmic-ray rejection | your 2D frames were **not** cleaned of cosmic rays: set `cr_clip` (e.g. 5) |
| 6 | 1D error calibration | `calibrate_errors` | rescale so residuals about a 41-pixel running median have unit scatter | you need errors that are strictly per-pixel (then skip it and say so) |

`extract_mask` wraps these for a whole mask and uses two mask-wide numbers: the median object
row (centre for faint slits whose profile can't be fitted) and the median error scale.

## Before running on new data

- **Plane order and flags.** Open one extension and check which plane is which, and how bad
  pixels are marked. Pass `--flux-plane / --err-plane` to `scripts/extract_mask.py`.
- **Wavelength solution.** It is read from `CRVAL1`, `CDELT1` (or `CD1_1`) and `CRPIX1`.
  Non-linear solutions need your own `wavelength_axis`.
- **Look at the QA plots** (`--qa`). Automatic tracing fails for faint objects, objects at the
  slit edge and slits with two objects. `extraction_log.csv` lists which slits got a flat trace.
- **Don't borrow traces between slits** unless the slit is too faint to trace on its own.

## Why optimal extraction gains less than you might expect

For sky-dominated faint galaxies the gain of Horne weighting over a well-placed boxcar is
small (a few per cent in S/N in our tests); most of the improvement over our original
extraction came from tracing each slit and masking bad pixels. Optimal extraction does recover
line fluxes more completely (~93% vs ~80% for a ±1.5σ box), which matters if you use them.

## Preparing spectra for Marz

Marz opens one file at a time. `scripts/pack_marz.py` puts any number of 1D spectra into one
multi-spectrum file:

```bash
python scripts/pack_marz.py all.fits spectra/*.fits --clean
```

- All spectra are rebinned (flux-conserving) onto one grid: by default the coarsest input
  sampling. Rebinning correlates neighbouring pixels.
- Marz runs its automatic fit only if a variance is present; spectra without an `ERR`
  extension get one estimated from their own scatter.
- `--clean` blanks the windows in `clusterspec/clean.py` (or your own list with `--windows`),
  the strongest sky lines, and dips/single-pixel spikes beyond 5σ. Without it, Marz tends to fit
  the telluric A/B bands as absorption lines at z ≈ 0.17. Note the default list also blanks the
  7170–7350 Å water band, which hides H-beta at z ≈ 0.47–0.51.
- Marz assumes air wavelengths unless the header says `VACUUM = T`.

### Comparing two versions of the same spectra blind

To test whether a new extraction is really better, measure both versions without knowing
which is which:

```bash
python scripts/pack_marz.py blind.fits old/*.fits new/*.fits --clean --blind key.csv
# measure everything in Marz, save the results, then:
python scripts/score_blind.py blind_results.mz key.csv --version-from-path old,new
```

Write down the rule for adopting the new version before you start measuring. The automatic QOP
alone is not a fair judge: check the fits by eye.
