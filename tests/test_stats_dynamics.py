import numpy as np
from clusterspec.stats import (biweight_location, biweight_scale, biweight_location_v,
                               biweight_scale_v, jackknife)
from clusterspec.dynamics import (projected_distance, m200_from_sigma, r200_from_m200,
                                  peculiar_velocity, dynamics)
from clusterspec.cosmo import PLANCK18, Cosmology


def test_biweight_gaussian():
    x = np.random.default_rng(1).normal(3.0, 2.0, 20000)
    assert abs(biweight_location(x) - 3.0) < 0.05
    assert abs(biweight_scale(x) - 2.0) < 0.05


def test_biweight_resists_outliers():
    x = np.r_[np.random.default_rng(2).normal(0, 1, 200), np.full(20, 50.0)]
    assert abs(biweight_location(x)) < 0.2 and biweight_scale(x) < 1.3


def test_vectorised_matches_scalar():
    X = np.random.default_rng(3).normal(0, 1, (5, 37))
    assert np.allclose(biweight_location_v(X), [biweight_location(r) for r in X])
    assert np.allclose(biweight_scale_v(X), [biweight_scale(r) for r in X])


def test_jackknife_mean():
    x = np.random.default_rng(4).normal(0, 1, 100)
    est, bias, se = jackknife(x, np.mean)
    assert abs(bias) < 1e-12 and abs(se - x.std(ddof=1) / 10) < 1e-12


def test_distances_and_mass():
    # 1 arcmin at z = 0.5 is ~0.37 physical Mpc for Planck18
    r = projected_distance(150.0, 2.0 + 1 / 60, 150.0, 2.0, 0.5)
    assert 0.36 < r < 0.39
    m = m200_from_sigma(1000.0, 0.3)
    assert 5e14 < m < 1.2e15
    assert 1.2 < r200_from_m200(m, 0.3) < 2.0
    assert np.isclose(peculiar_velocity(0.31, 0.30), 299792.458 * 0.01 / 1.3)


def test_curved_cosmology_runs():
    c = Cosmology(70, 0.3, 0.6)
    assert c.angular_diameter_distance(0.5) > 0


def test_dynamics_recovers_sigma():
    rng = np.random.default_rng(5)
    z = 0.4 + rng.normal(0, 800, 400) * 1.4 / 299792.458
    d = dynamics(z)
    assert abs(d['sigma_v'] - 800) < 60 and abs(d['z'] - 0.4) < 1e-3
