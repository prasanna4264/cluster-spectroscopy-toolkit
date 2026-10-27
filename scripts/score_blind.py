#!/usr/bin/env python
"""Score a blind comparison of two sets of spectra of the same objects (e.g. two extractions).

    python scripts/score_blind.py marz_results.mz key.csv --version-from-path old,new

key.csv comes from ``pack_marz.py --blind``. Each input file must belong to one version,
identified by a substring of its path (e.g. folders old/ and new/), and the same object must
have the same file name in both. Reports, per version, how many spectra reach QOP >= 2 and
>= 3, and how well the redshifts agree where both versions succeed.

Decide your adoption rule BEFORE measuring, e.g. "adopt the new version if it gives >= 10 more
QOP >= 3 redshifts and >= 95% of objects with QOP >= 3 in both agree within 300 km/s".
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from clusterspec.marz import read_marz_results, telluric_pileup       # noqa: E402

C = 299792.458


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('marz')
    ap.add_argument('key')
    ap.add_argument('--version-from-path', required=True, help='comma-separated substrings, e.g. old,new')
    ap.add_argument('--dv', type=float, default=300.0)
    a = ap.parse_args()
    tags = a.version_from_path.split(',')
    key = pd.read_csv(a.key)
    key['version'] = key.file.map(lambda p: next((t for t in tags if t in p), None))
    key['object'] = key.file.map(os.path.basename)
    res = read_marz_results(a.marz)
    m = key.merge(res, on='name', how='left')
    m['qop'] = m.qop.fillna(0).astype(int)
    w = m.pivot_table(index='object', columns='version', values=['z', 'qop'], aggfunc='first')
    w.columns = [f'{x}_{y}' for x, y in w.columns]
    A, B = tags[0], tags[1]
    for t in tags:
        print(f"{t:>8}: QOP>=2 {int((w[f'qop_{t}'] >= 2).sum()):4d}   QOP>=3 {int((w[f'qop_{t}'] >= 3).sum()):4d}"
              f"   QOP>=3 at z 0.14-0.20 (telluric?) {telluric_pileup(w[f'z_{t}'], w[f'qop_{t}'])}")
    both = w[(w[f'qop_{A}'] >= 3) & (w[f'qop_{B}'] >= 3)]
    dv = C * (both[f'z_{B}'] - both[f'z_{A}']).abs() / (1 + both[f'z_{A}'])
    gain = int(((w[f'qop_{B}'] >= 3) & (w[f'qop_{A}'] < 3)).sum() - ((w[f'qop_{A}'] >= 3) & (w[f'qop_{B}'] < 3)).sum())
    print(f'\nnet QOP>=3 gain of {B} over {A}: {gain}')
    if len(both):
        print(f'QOP>=3 in both: {len(both)}, |dv| < {a.dv:.0f} km/s: {(dv < a.dv).sum()} ({(dv < a.dv).mean():.0%})')
    out = os.path.splitext(a.marz)[0] + '_scored.csv'
    w.to_csv(out)
    print(f'per-object table -> {out}')


if __name__ == '__main__':
    main()
