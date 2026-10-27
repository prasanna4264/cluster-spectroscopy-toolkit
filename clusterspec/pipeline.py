"""One-call analysis of a cluster field, and a loop over a catalogue of clusters."""
import numpy as np
import pandas as pd
from .members import select_members, seed_redshift
from .dynamics import (dynamics, m200_from_sigma, r200_from_m200, peculiar_velocity,
                       projected_distance)
from .ds import ds_test, default_n_neighbours
from .cosmo import PLANCK18


def analyse_cluster(z, ra, dec, z_seed=None, ra0=None, dec0=None, mode='velocity',
                    dv_max=5000.0, n_min_dynamics=15, n_min_ds=20, n_mc=1000, seed=0,
                    cosmo=PLANCK18, alpha=0.05, **aperture_kw):
    """Membership, dynamics and DS test for one cluster field.

    z, ra, dec    : all galaxies with redshifts in the field (degrees for positions).
    z_seed        : first guess of the cluster redshift (e.g. a published value). If None,
                    estimated from the field with ``seed_redshift`` (and the centre too, if
                    ra0/dec0 are None).
    mode          : membership: 'velocity', 'fixed' (pass r_fixed=...) or 'r200'.
    Returns a dict with summary numbers plus arrays (members mask, radii, velocities, DS
    Monte Carlo) for plotting. Fields that fail a minimum-N cut return status != 'ok'."""
    z, ra, dec = (np.asarray(a, float) for a in (z, ra, dec))
    out = dict(mode=mode, N_field=z.size)
    src = 'given'
    if z_seed is None or ra0 is None or dec0 is None:
        s = seed_redshift(z, ra, dec)
        if s is None:
            return dict(out, status='no_seed')
        z_seed = s[0] if z_seed is None else z_seed
        if ra0 is None or dec0 is None:
            ra0, dec0 = s[1], s[2]
        src = 'estimated'
    out.update(z_seed=z_seed, ra0=ra0, dec0=dec0, seed_source=src)
    mem, info = select_members(z, ra, dec, z_seed, ra0, dec0, mode=mode, dv_max=dv_max,
                               cosmo=cosmo, **aperture_kw)
    out.update(info)
    N = int(mem.sum())
    out['N_members'] = N
    if N < n_min_dynamics:
        return dict(out, status=f'N<{n_min_dynamics}', members=mem)
    d = dynamics(z[mem])
    m200 = m200_from_sigma(d['sigma_v'], d['z'], cosmo)
    out.update(z_cl=d['z'], z_cl_err=d['z_err'], z_ci_lo=d['z_ci'][0], z_ci_hi=d['z_ci'][1],
               sigma_v=d['sigma_v'], sigma_v_err=d['sigma_v_err'], M200=m200,
               R200=r200_from_m200(m200, d['z'], cosmo))
    out['members'] = mem
    out['R'] = projected_distance(ra, dec, ra0, dec0, d['z'], cosmo)
    out['v_pec'] = peculiar_velocity(z, d['z'])
    if N >= n_min_ds:
        v = out['v_pec'][mem]
        t = ds_test(v, z[mem], ra[mem], dec[mem], d['sigma_v'], n_mc=n_mc, seed=seed,
                    n_nn=default_n_neighbours(N))
        out.update(ds_delta=t['delta_obs'], ds_p=t['p_value'],
                   ds_substructure=bool(t['p_value'] <= alpha), ds=t)
    else:
        out.update(ds_delta=np.nan, ds_p=np.nan, ds_substructure=None)
    return dict(out, status='ok')


SUMMARY_COLS = ['cluster', 'status', 'mode', 'N_field', 'N_members', 'z_seed', 'seed_source',
                'z_cl', 'z_cl_err', 'z_ci_lo', 'z_ci_hi', 'sigma_v', 'sigma_v_err', 'M200',
                'R200', 'radius', 'converged', 'ds_delta', 'ds_p', 'ds_substructure']


def run_catalog(catalog, centres=None, cluster_col='cluster', z_col='z', ra_col='ra',
                dec_col='dec', verbose=True, **kw):
    """Run ``analyse_cluster`` on every cluster in a galaxy catalogue.

    catalog : DataFrame with one row per galaxy (cluster name, z, ra, dec).
    centres : optional DataFrame indexed by cluster name with columns z, ra, dec
              (published redshift and centre). Missing clusters are seeded from the data.
    Returns (summary DataFrame, dict of full per-cluster results)."""
    rows, full = [], {}
    for name, g in catalog.groupby(cluster_col, sort=True):
        c = centres.loc[name] if centres is not None and name in centres.index else None
        r = analyse_cluster(g[z_col].values, g[ra_col].values, g[dec_col].values,
                            z_seed=None if c is None else c['z'],
                            ra0=None if c is None else c['ra'],
                            dec0=None if c is None else c['dec'], **kw)
        r['cluster'] = name
        full[name] = r
        rows.append({k: r.get(k, np.nan) for k in SUMMARY_COLS})
        if verbose:
            msg = (f"N={r['N_members']:3d}  sigma_v={r['sigma_v']:6.0f}  p_DS={r['ds_p']:.3f}"
                   if r['status'] == 'ok' else r['status'])
            print(f'{name:28s} {msg}')
    return pd.DataFrame(rows), full
