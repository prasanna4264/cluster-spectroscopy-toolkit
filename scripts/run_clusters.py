#!/usr/bin/env python
"""Membership, dynamics and DS substructure test for every cluster in a galaxy catalogue.

    python scripts/run_clusters.py galaxies.csv results.csv [--centres centres.csv]
                                   [--mode velocity|fixed|r200] [--r-fixed 1.5]

galaxies.csv : one row per galaxy with columns cluster, z, ra, dec (degrees); other names
               can be given with --cols cluster,z,ra,dec
centres.csv  : optional; columns cluster, z, ra, dec with published redshifts and centres.
               Clusters missing from it are seeded from the data (redshift-histogram peak,
               median position of galaxies near it).
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from clusterspec import run_catalog                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('catalog')
    ap.add_argument('output')
    ap.add_argument('--centres')
    ap.add_argument('--mode', default='velocity', choices=['velocity', 'fixed', 'r200'])
    ap.add_argument('--r-fixed', type=float, default=1.5)
    ap.add_argument('--dv', type=float, default=5000.0, help='initial velocity window (km/s)')
    ap.add_argument('--n-mc', type=int, default=1000)
    ap.add_argument('--cols', default='cluster,z,ra,dec')
    a = ap.parse_args()

    cc, zc, rc, dc = a.cols.split(',')
    cat = pd.read_csv(a.catalog).dropna(subset=[zc, rc, dc])
    cen = None
    if a.centres:
        cen = pd.read_csv(a.centres).set_index('cluster')[['z', 'ra', 'dec']]
    kw = dict(mode=a.mode, dv_max=a.dv, n_mc=a.n_mc)
    if a.mode == 'fixed':
        kw['r_fixed'] = a.r_fixed
    summary, _ = run_catalog(cat, cen, cluster_col=cc, z_col=zc, ra_col=rc, dec_col=dc, **kw)
    summary.to_csv(a.output, index=False)
    ok = summary[summary.status == 'ok']
    ds = ok.dropna(subset=['ds_p'])
    print(f'\n{len(ok)} clusters analysed; DS run on {len(ds)}, '
          f'{int((ds.ds_p <= 0.05).sum())} with p <= 0.05 -> {a.output}')


if __name__ == '__main__':
    main()
