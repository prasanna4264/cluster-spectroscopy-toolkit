# cluster-spectroscopy-toolkit

Tools for going from multi-slit spectra to galaxy-cluster kinematics:

1. **Extraction**: 1D spectra from rectified 2D multi-slit data (aperture or Horne optimal
   extraction), with error calibration
2. **Redshifts**: cleaning spectra before template fitting, and packing hundreds of spectra
   into one [Marz](https://samreay.github.io/Marz/) file, including a blind mode for comparing
   two versions of the same spectra
3. **Cluster analysis**: membership (velocity-only, fixed aperture or iterated R200),
   biweight redshift and velocity dispersion, M200, and the Dressler–Shectman substructure test
   with a mock-based check of its detection power

I wrote this for Magellan/IMACS spectroscopy of strong-lensing galaxy clusters from the SDSS
Giant Arcs Survey (Adhikari et al., in prep.), but nothing in it is specific to that
instrument or sample. No data are included; `examples/make_mock_data.py` makes mock data so
everything can be run straight away.

Only numpy, scipy, pandas and matplotlib are required. astropy is used for reading FITS files
if it's installed, but there is a small built-in reader/writer as well.

## Quick start

```bash
git clone https://github.com/prasanna4264/cluster-spectroscopy-toolkit.git
cd cluster-spectroscopy-toolkit
pip install -e .                      # or just run the scripts from the repo folder

python examples/make_mock_data.py     # mock 2D mask + mock cluster catalogue

# 1. extract every slitlet of a mask
python scripts/extract_mask.py examples/data/mock_mask_2spec.fits out/spectra --qa

# 2. one Marz file for all of them (telluric bands, strong sky lines and spikes masked)
python scripts/pack_marz.py out/marz_all.fits out/spectra/*.fits --clean

# 3. membership, dynamics and DS test for every cluster in a catalogue
python scripts/run_clusters.py examples/data/mock_galaxies.csv out/clusters.csv \
       --centres examples/data/mock_centres.csv --mode r200
```

From Python:

```python
from clusterspec import analyse_cluster
res = analyse_cluster(z, ra, dec, z_seed=0.45, ra0=150.1, dec0=2.2, mode='r200')
print(res['N_members'], res['sigma_v'], res['M200'], res['ds_p'])

from clusterspec import plots
plots.phase_space(res)
```

## What's in here

| path | what |
|---|---|
| `clusterspec/extract.py` | bad-pixel and chip-gap masks, spatial profile, trace, aperture and optimal extraction, error calibration, whole-mask driver |
| `clusterspec/clean.py` | telluric / sky-line windows, sky-line columns, chip-gap zero runs, outlier dips and single-pixel spikes |
| `clusterspec/marz.py` | flux-conserving rebinning, multi-spectrum Marz files, reading Marz results |
| `clusterspec/members.py` | velocity window, biweight sigma clipping, redshift seeding, fixed and R200 apertures |
| `clusterspec/dynamics.py` | biweight z and sigma_v with jackknife errors, M200 (Evrard et al. 2008), R200, projected radii |
| `clusterspec/ds.py` | Dressler & Shectman (1988) test, vectorised Monte Carlo |
| `clusterspec/power.py` | DS false-positive rate and detection power on a cluster's real positions |
| `clusterspec/pipeline.py` | `analyse_cluster` and `run_catalog` |
| `scripts/` | command-line versions: `extract_mask.py`, `pack_marz.py`, `run_clusters.py`, `score_blind.py` |
| `docs/` | [extraction](docs/extraction.md) and [analysis](docs/analysis.md) notes: every step, and what you may need to change for your data |
| `tests/` | `pytest`, or `python tests/run_tests.py` without it |

## Things I learned the hard way

- **Trace every slit separately.** Objects don't sit at the same height in every slitlet.
  Borrowing one slit's trace for a whole mask put the aperture 2+ pixels off for about a quarter
  of the slits on one of our masks; 4 pixels off loses more than half the flux.
- **Check the error image.** Ours underestimated the real scatter by 12–16% per pixel, and by
  ~50% in 1D because rectified pixels are correlated. `extract.py` rescales both.
- **Mask the telluric bands before automatic redshift fitting.** Marz fitted the A and B bands
  as H-alpha and Na D absorption at z ≈ 0.17 for dozens of faint spectra.
- **Velocity-only membership can reach several Mpc** in widely surveyed fields, and infall
  regions or neighbouring groups out there can produce a substructure signal on their own.
  Run the R200 aperture as a check.
- **Know the power of your substructure test.** At N ≈ 20–30 the DS test misses most real
  mergers; `power.py` tells you how many it would catch for your cluster's geometry.

## License

MIT (see `LICENSE`). If this is useful for a paper, a link to the repository is appreciated.
