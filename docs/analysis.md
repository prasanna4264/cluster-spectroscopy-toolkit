# Cluster membership, dynamics and substructure

`analyse_cluster(z, ra, dec, z_seed, ra0, dec0, mode=...)` runs the steps below on one field.
`run_catalog` loops over a catalogue; `scripts/run_clusters.py` is the command-line version.

## 1. Seed redshift and centre

Use published values where they exist (`--centres`). Otherwise `seed_redshift` takes the
peak of a smoothed redshift histogram (0.01 bins, 0.02 smoothing), refines it with a biweight
within ±5000 km/s, and uses the median position of those galaxies as the centre. The seed only
sets the initial velocity window; the final cluster redshift comes from the members.

## 2. Membership

1. **Velocity window**: |Δv| < 5000 km/s (rest frame) around the seed (`--dv`). Removes
   foreground galaxies and lensed background sources before clipping.
2. **Biweight clipping**: drop galaxies outside location ± 3 × scale (computed in redshift),
   repeat until nothing changes.
3. **Aperture** (optional, `mode`):
   - `velocity`: no spatial cut. Classical, but in widely surveyed fields members can extend
     to 4–6 Mpc.
   - `fixed`: keep R < `r_fixed` Mpc (default 1.5) before clipping.
   - `r200`: start at 1.5 Mpc; clip → σ_v → M200 → R200 → new radius; repeat until R200 changes
     by < 0.01 Mpc. Each step changes the radius by at most ×1.5, and if the radius flips between
     two values (a few edge galaxies entering and leaving) the smaller one is used. The `converged`
     column says which happened (`yes`, `cycle_min`, `too_few`, `no`).

R200 from σ_v is somewhat circular (contaminated σ_v → larger R200 → more contamination). If you
have an independent mass (X-ray, SZ, lensing), use its R200 with `mode='fixed'`.

## 3. Dynamics

- z_cl and σ_v: biweight location and scale of the members (Beers et al. 1990); σ_v =
  c × scale / (1 + z_cl); jackknife errors; 95% interval on z_cl centred on the biweight value.
- M200 from the Evrard et al. (2008) dark-matter relation, σ = 1082.9 (h(z) M200 / 10^15)^0.3361
  km/s. No galaxy velocity bias is applied.
- R200 for 200 × the critical density at z_cl.
- Cosmology: flat Planck 2018 H0 and Ωm by default; pass `cosmo=Cosmology(...)` to change it.

Treat σ_v > ~2000 km/s with suspicion: usually two structures along the line of sight.

## 4. Dressler–Shectman test

For each member i, with its n nearest neighbours on the sky (n = max(8, √N); group of n + 1
including itself):

δᵢ² = ((n+1) / σ²) [ (v̄_local − v̄)² + (σ_local − σ)² ]

Δ = Σ δᵢ; the statistic is Δ/N. The p-value is the fraction of 1000 shuffles of the velocities
(keeping positions fixed) with Δ/N at least as large as observed. Conventions, in case you
compare with other codes: local mean is a plain mean, local and global dispersions are biweight
scales, global mean is the biweight location, and (v, z) are shuffled together. The Monte Carlo
uses a fixed seed so results are reproducible.

The test is only run for N ≥ 20 (`n_min_ds`). Below ~30 members it has little power.

## 5. Is a non-detection meaningful? (`power.py`)

`ds_power(ra, dec, sigma, z_cl, frac_sub, dv)` keeps a cluster's real member positions and
draws mock velocities: Gaussian (null) or with a compact subgroup (`frac_sub` of the members
nearest a random galaxy, offset by `dv`·σ along the line of sight). It returns the fraction of
mocks with p ≤ 0.05. The null rate should come out near 5%; the merger rate is the power.
Expect low power for small clusters: a single non-detection at N ≈ 20–30 says little.

## Checks worth doing on any sample

- Compare `velocity` and `r200` membership. If substructure detections disappear inside R200,
  they came from the outskirts or from neighbouring structures, not the cluster core.
- Look for exact duplicate positions in your catalogue (the same galaxy entered twice). They
  change nearest-neighbour groups and bias the DS test.
- Plot the phase-space diagram (`plots.phase_space`) for every cluster before trusting σ_v.
