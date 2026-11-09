"""Instanzen und Szenarien: Determinismus, Grenzen, Verteilungseigenschaften (Bänder, keine exakten Gleitkommawerte)."""

import numpy as np
import pytest

import bsa_constants as C
import bsa_scenario as S


def test_instance_is_deterministic():
    assert S.make_instance(4, 0.3, 0.5, 0.75, C.DIST_NORMAL, 11) == S.make_instance(4, 0.3, 0.5, 0.75, C.DIST_NORMAL, 11)
    assert S.make_instance(4, 0.3, 0.5, 0.75, C.DIST_NORMAL, 11).mu != S.make_instance(4, 0.3, 0.5, 0.75, C.DIST_NORMAL, 12).mu


def test_spread_zero_gives_equal_warehouses():
    inst = S.make_instance(5, 0.2, 0.0, 0.75, C.DIST_NORMAL, 3)
    assert set(inst.mu) == {C.MU_BASE} and set(inst.sd) == {C.CV_NORMAL * C.MU_BASE}


def test_sizes_stay_inside_the_spread_band():
    for seed in range(30):
        inst = S.make_instance(6, 0.2, 0.9, 0.75, C.DIST_NORMAL, seed)
        assert all(C.MU_BASE * 0.1 <= m <= C.MU_BASE * 1.9 for m in inst.mu)


def test_cu_plus_co_follows_the_service_level():
    assert S.make_instance(2, 0, 0, 0.75, C.DIST_NORMAL, 1).cu_plus_co == pytest.approx(4.0)     # c_u = 3, c_o = 1
    assert S.make_instance(2, 0, 0, 0.5, C.DIST_NORMAL, 1).cu_plus_co == pytest.approx(2.0)


@pytest.mark.parametrize("kwargs", [dict(dist="Gleichverteilung"), dict(service=1.0), dict(service=0.0), dict(rho=1.0), dict(rho=-0.1), dict(spread=1.0)])
def test_invalid_parameters_raise(kwargs):
    params = dict(n=3, rho=0.2, spread=0.2, service=0.75, dist=C.DIST_NORMAL, seed=1)
    params.update(kwargs)
    with pytest.raises(ValueError):
        S.make_instance(**params)


def test_covariance_matrix_structure():
    inst = S.make_instance(3, 0.4, 0.5, 0.75, C.DIST_NORMAL, 8)
    cov = S.covariance(inst)
    sd = np.array(inst.sd)
    assert np.allclose(np.diag(cov), sd**2)
    assert cov[0, 1] == pytest.approx(0.4 * sd[0] * sd[1])
    assert np.allclose(cov, cov.T) and np.all(np.linalg.eigvalsh(cov) > 0)


def test_standard_normals_have_the_requested_correlation_in_a_band():
    z = S.standard_normals(3, 0.6, 4)
    corr = np.corrcoef(z, rowvar=False)
    assert z.shape == (C.SCENARIOS, 3)
    assert abs(corr[0, 1] - 0.6) < 0.03 and abs(corr[1, 2] - 0.6) < 0.03
    assert abs(z.mean()) < 0.03 and abs(z.std() - 1.0) < 0.03


def test_scenarios_are_reproducible_and_depend_on_the_seed():
    a = S.scenario_demands(S.make_instance(3, 0.3, 0.5, 0.75, C.DIST_LOGNORMAL, 5), 200)
    b = S.scenario_demands(S.make_instance(3, 0.3, 0.5, 0.75, C.DIST_LOGNORMAL, 5), 200)
    c = S.scenario_demands(S.make_instance(3, 0.3, 0.5, 0.75, C.DIST_LOGNORMAL, 6), 200)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


@pytest.mark.parametrize("dist", C.DISTS)
def test_scenario_means_are_near_mu_in_a_band(dist):
    """Mittel der Szenarien nahe mu (Normal und Lognormal: 3 %; Pareto mit Index 2,5 hat einen schweren Tail: 10 %)."""
    inst = S.make_instance(3, 0.0, 0.5, 0.75, dist, 21)
    d = S.scenario_demands(inst)
    band = 0.10 if dist == C.DIST_HEAVY else 0.03
    assert np.all(np.abs(d.mean(axis=0) / np.array(inst.mu) - 1.0) < band)


def test_heavy_tail_is_heavier_than_lognormal_in_the_far_tail():
    """Verhältnis des 99,9-%-Quantils zum Mittel: bei Pareto (Index 2,5) größer als bei Lognormal (Form 0,5), gemessen an einem Lager mit rho = 0."""
    ratios = {}
    for dist in (C.DIST_LOGNORMAL, C.DIST_HEAVY):
        d = S.scenario_demands(S.make_instance(2, 0.0, 0.0, 0.75, dist, 2))[:, 0]
        ratios[dist] = np.quantile(d, 0.999) / d.mean()
    assert ratios[C.DIST_HEAVY] > ratios[C.DIST_LOGNORMAL]


def test_normal_scenarios_use_the_instance_sd():
    inst = S.make_instance(2, 0.0, 0.5, 0.75, C.DIST_NORMAL, 4)
    d = S.scenario_demands(inst)
    assert np.all(np.abs(d.std(axis=0) / np.array(inst.sd) - 1.0) < 0.03)


def test_sd_is_cv_times_mu_times_a_factor_inside_the_documented_band():
    """sd_i = 0,3 * mu_i * Faktor mit Faktor in [1 - 0,5 * Streuung, 1 + 0,5 * Streuung] (Text der Streuungs-Hilfe in der App)."""
    for spread in (0.2, 0.9):
        for seed in range(20):
            inst = S.make_instance(5, 0.1, spread, 0.75, C.DIST_NORMAL, seed)
            factor = np.array(inst.sd) / (C.CV_NORMAL * np.array(inst.mu))
            assert np.all(factor >= 1 - 0.5 * spread - 1e-12) and np.all(factor <= 1 + 0.5 * spread + 1e-12)
