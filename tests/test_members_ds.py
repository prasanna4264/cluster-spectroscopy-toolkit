import numpy as np
from clusterspec.members import select_members, sigma_clip_members, seed_redshift
from clusterspec.ds import ds_test, neighbour_table
from clusterspec.power import ds_power
from clusterspec.pipeline import analyse_cluster

C = 299792.458


def mock(n=60, z0=0.4, sigma=900, seed=0, n_field=30):
    rng = np.random.default_rng(seed)
    ra = 150 + rng.normal(0, 0.02, n)
    dec = 2 + rng.normal(0, 0.02, n)
    z = z0 + rng.normal(0, sigma, n) * (1 + z0) / C
    zf = rng.uniform(0.05, 1.5, n_field)
    return (np.r_[z, zf], np.r_[ra, 150 + rng.uniform(-.1, .1, n_field)],
            np.r_[dec, 2 + rng.uniform(-.1, .1, n_field)], n)


def test_clipping_removes_interlopers():
    z, ra, dec, n = mock()
    m, info = select_members(z, ra, dec, 0.4, 150, 2, mode='velocity')
    assert m[:n].mean() > 0.95 and m[n:].sum() <= 2


def test_r200_converges():
    z, ra, dec, n = mock(n=80)
    m, info = select_members(z, ra, dec, 0.4, 150, 2, mode='r200')
    assert info['converged'] in ('yes', 'cycle_min') and 1.0 < info['radius'] < 2.5


def test_seed_redshift():
    z, ra, dec, n = mock()
    zs, rc, dc = seed_redshift(z, ra, dec)
    assert abs(zs - 0.4) < 0.005 and abs(rc - 150) < 0.02


def test_neighbour_table_self_first_with_duplicates():
    ra = np.array([1.0, 1.0, 1.001, 1.002, 1.003, 1.004])
    dec = np.zeros(6)
    idx = neighbour_table(ra, dec, 3)
    assert all(idx[i, 0] == i for i in range(6))


def test_ds_null_and_merger():
    rng = np.random.default_rng(7)
    n = 60
    ra, dec = 150 + rng.normal(0, .02, n), 2 + rng.normal(0, .02, n)
    v = rng.normal(0, 900, n)
    p_null = ds_test(v, 0.4 + v * 1.4 / C, ra, dec, 900, n_mc=300)['p_value']
    # move the 15 galaxies nearest one point by +2 sigma
    d = (ra - ra[0]) ** 2 + (dec - dec[0]) ** 2
    g = np.argsort(d)[:15]
    v2 = v.copy()
    v2[g] = rng.normal(1800, 300, 15)
    p_mer = ds_test(v2, 0.4 + v2 * 1.4 / C, ra, dec, 900, n_mc=300)['p_value']
    assert p_null > 0.05 and p_mer < 0.01


def test_power_false_positive_rate():
    rng = np.random.default_rng(8)
    ra, dec = 150 + rng.normal(0, .02, 40), 2 + rng.normal(0, .02, 40)
    fp = ds_power(ra, dec, 900, 0.4, frac_sub=0, n_mock=60, n_mc=150)
    assert fp < 0.15


def test_analyse_cluster_end_to_end():
    z, ra, dec, n = mock(n=50)
    r = analyse_cluster(z, ra, dec, 0.4, 150, 2, n_mc=200)
    assert r['status'] == 'ok' and abs(r['sigma_v'] - 900) < 250 and 0 <= r['ds_p'] <= 1
