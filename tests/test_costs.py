"""Einheiten der Koalitionskosten an Mini-Beispielen mit Handrechnung."""

import numpy as np
import pytest

import bsa_costs as K
import bsa_constants as C
from bsa_scenario import make_instance
from conftest import newsvendor_k


def test_mask_matrix_bits():
    m = K.mask_matrix(3)
    assert m.shape == (8, 3)
    assert m[0].tolist() == [0, 0, 0]
    assert m[5].tolist() == [1, 0, 1]          # 5 = 0b101: Lager 1 und Lager 3
    assert m[7].tolist() == [1, 1, 1]


def test_normal_cost_is_k_times_sigma():
    """C = k * sigma mit k = phi(z) / (1 - Servicegrad), k über scipy.stats gerechnet (hier wird die Formel der Demo nachgerechnet)."""
    for service in (0.5, 0.75, 0.9, 0.99):
        assert K.normal_newsvendor_cost(7.0, service) == pytest.approx(7.0 * newsvendor_k(service), rel=1e-12)


def test_service_half_costs_are_symmetric():
    """Servicegrad 0,5: c_u = c_o = 1, z = 0, C = 2 * phi(0) * sigma = sigma * sqrt(2/pi)."""
    assert K.normal_newsvendor_cost(1.0, 0.5) == pytest.approx(np.sqrt(2.0 / np.pi), rel=1e-12)


def test_coalition_costs_cov_two_members():
    """sigma = 3 und 4, unabhängig: C({1}) = 3k, C({2}) = 4k, C(N) = 5k; die leere Koalition kostet 0."""
    k = newsvendor_k(0.75)
    c = K.coalition_costs_cov(np.diag([9.0, 16.0]), 0.75)
    assert c.tolist() == pytest.approx([0.0, 3 * k, 4 * k, 5 * k], rel=1e-12)


def test_coalition_costs_cov_perfect_correlation_is_additive():
    """rho = 1: sigma_N = sigma_1 + sigma_2, Bündeln spart nichts."""
    cov = np.outer([3.0, 4.0], [3.0, 4.0])
    c = K.coalition_costs_cov(cov, 0.75)
    assert c[3] == pytest.approx(c[1] + c[2], rel=1e-9)


def test_empirical_cost_hand_example():
    """Eine Spalte mit den Szenarien 1, 2, 3, 4 und Servicegrad 0,75: c_u = 3, c_o = 1, Bestand q = 3 (Index ceil(4 * 0,75) - 1 = 2).
    Kosten = (3 * (4 - 3) + 1 * ((3 - 1) + (3 - 2))) / 4 = 6 / 4 = 1,5."""
    totals = np.array([[3.0], [1.0], [4.0], [2.0]])                       # unsortiert
    assert K.empirical_newsvendor_cost(totals, 0.75)[0] == pytest.approx(1.5)


def test_empirical_cost_is_minimum_over_all_order_quantities():
    totals = np.array([[5.0], [1.0], [9.0], [3.0], [7.0], [2.0]])
    best = K.empirical_newsvendor_cost(totals, 0.6)[0]
    cu, co = 0.6 / 0.4, 1.0
    t = totals[:, 0]
    grid = [np.mean(cu * np.maximum(t - q, 0) + co * np.maximum(q - t, 0)) for q in np.linspace(0, 10, 2001)]
    assert best == pytest.approx(min(grid), abs=1e-3)


def test_scenario_costs_match_a_plain_loop():
    """Die vektorisierte Fassung gegen eine Schleife über Koalitionen (anderer Rechenweg, kleine Szenarienzahl)."""
    inst = make_instance(3, 0.4, 0.5, 0.8, C.DIST_LOGNORMAL, 5)
    from bsa_scenario import scenario_demands

    d = scenario_demands(inst, 500)
    fast = K.coalition_costs_scenarios(d, 0.8, block=3)
    for s in range(1, 8):
        t = sum(d[:, i] for i in range(3) if s >> i & 1)
        q = np.sort(t)[int(np.ceil(500 * 0.8)) - 1]
        cost = np.mean(4.0 * np.maximum(t - q, 0) + np.maximum(q - t, 0))        # c_u = 0,8 / 0,2 = 4
        assert fast[s] == pytest.approx(cost, rel=1e-9)
    assert fast[0] == 0.0


def test_costs_of_a_single_member_are_subadditive_in_coalitions():
    """Bündeln kostet nie mehr als getrennt bestellen (C(S + T) <= C(S) + C(T)), hier für die Gleichkorrelation ohne und mit hohem rho."""
    for rho in (0.0, 0.5, 0.95):
        c = K.coalition_costs_normal(make_instance(4, rho, 0.6, 0.8, C.DIST_NORMAL, 9))
        for s in range(1, 16):
            for t in range(1, 16):
                if s & t == 0:
                    assert c[s | t] <= c[s] + c[t] + 1e-12
