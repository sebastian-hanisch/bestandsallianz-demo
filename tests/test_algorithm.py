"""Kern gegen Handrechnung an den Mini-Spielen aus conftest.py (DEMO-PLAYBOOK §1b: jede Einheit einzeln)."""

import numpy as np
import pytest

import bsa_algorithm as A
import bsa_constants as C
from conftest import make_cost_game


def test_make_game_checks_length_and_empty_coalition():
    with pytest.raises(ValueError):
        A.make_game([0.0, 1.0, 2.0])                    # Länge 3 ist keine Zweierpotenz
    with pytest.raises(ValueError):
        A.make_game([1.0, 1.0])                         # die leere Koalition kostet nichts
    assert A.make_game([0.0, 4.0]).n == 1


def test_grand_and_single_costs(mini):
    game, k = mini
    assert game.grand == pytest.approx(5 * k)
    assert game.single.tolist() == pytest.approx([3 * k, 4 * k])


def test_gain_of_pooling(mini):
    game, _ = mini
    assert A.gain_of_pooling(game) == pytest.approx(1 - 5 / 7)


def test_gain_without_costs_is_zero():
    assert A.gain_of_pooling(A.make_game([0.0, 0.0, 0.0, 0.0])) == 0.0


def test_shapley_two_members_by_hand(mini):
    game, k = mini
    assert A.shapley_value(game).tolist() == pytest.approx([2 * k, 3 * k], rel=1e-12)


def test_shapley_is_efficient_and_symmetric_players_pay_equally(empty_core_game):
    phi = A.shapley_value(empty_core_game)
    assert phi.sum() == pytest.approx(16.0)
    assert phi.tolist() == pytest.approx([16 / 3] * 3)       # drei gleiche Lager: jedes zahlt ein Drittel


def test_shapley_three_members_hand_example():
    """C({i}) = 6, C({i,j}) = 9, C(N) = 12 (symmetrisch): Zusatzkosten 6, 3, 3 in den Positionen 1 bis 3, Mittel 4 je Lager."""
    game = make_cost_game({(0,): 6, (1,): 6, (2,): 6, (0, 1): 9, (0, 2): 9, (1, 2): 9, (0, 1, 2): 12}, 3)
    assert A.shapley_value(game).tolist() == pytest.approx([4.0, 4.0, 4.0])


def test_shapley_asymmetric_three_members_hand_example():
    """Lager 1 kostet allein 10, Lager 2 und 3 allein je 4; jede Zusammenlegung kostet nur den größten Einzelwert:
    C(12) = C(13) = 10, C(23) = 4, C(N) = 10. Zusatzkosten von Lager 1 je Reihenfolge: 123: 10, 132: 10, 213: 10 - 4 = 6, 312: 10 - 4 = 6,
    231: 10 - 4 = 6, 321: 10 - 4 = 6. Mittel = 44 / 6."""
    game = make_cost_game({(0,): 10, (1,): 4, (2,): 4, (0, 1): 10, (0, 2): 10, (1, 2): 4, (0, 1, 2): 10}, 3)
    phi = A.shapley_value(game)
    assert phi[0] == pytest.approx(44 / 6)
    assert phi.sum() == pytest.approx(10.0)
    assert phi[1] == pytest.approx(phi[2])                    # Lager 2 und 3 sind gleich


def test_allocation_rules_by_hand(mini):
    game, k = mini
    assert A.allocation_equal(game).tolist() == pytest.approx([2.5 * k, 2.5 * k])
    assert A.allocation_proportional(game, [3.0, 4.0]).tolist() == pytest.approx([5 * k * 3 / 7, 5 * k * 4 / 7])
    assert A.allocate(C.RULE_SINGLE, game, [100.0, 100.0]).tolist() == pytest.approx([5 * k * 3 / 7, 5 * k * 4 / 7])   # Einzelkosten 3k : 4k
    assert A.allocate(C.RULE_MEAN, game, [100.0, 300.0]).tolist() == pytest.approx([1.25 * k, 3.75 * k])
    assert A.allocate(C.RULE_BASELINE, game, [1.0, 1.0]).tolist() == pytest.approx([3 * k, 4 * k])


def test_allocate_rejects_unknown_rule(mini):
    with pytest.raises(ValueError):
        A.allocate("Losentscheid", mini[0], [1.0, 1.0])


def test_proportional_with_zero_weights_falls_back_to_equal(mini):
    game, _ = mini
    assert A.allocation_proportional(game, [0.0, 0.0]).tolist() == A.allocation_equal(game).tolist()


def test_excess_and_max_excess(mini):
    game, k = mini
    x = np.array([2 * k, 3 * k])                              # Shapley
    e = A.excess(game, x)
    assert e[1] == pytest.approx(-k) and e[2] == pytest.approx(-k) and e[3] == pytest.approx(0.0, abs=1e-12)
    value, mask = A.max_excess(game, x)
    assert value == pytest.approx(-k) and mask in (1, 2)       # Gleichstand: kleinste Maske
    assert mask == 1


def test_max_excess_single_member_has_no_proper_coalition():
    assert A.max_excess(A.make_game([0.0, 7.0]), [7.0]) == (0.0, 0)


def test_is_in_core(mini):
    game, k = mini
    assert A.is_in_core(game, [2 * k, 3 * k])
    assert A.is_in_core(game, [k, 4 * k])                     # Rand des Kerns: x_1 = k, x_2 = 4k = C({2})
    assert not A.is_in_core(game, [0.9 * k, 4.1 * k])         # Lager 2 zahlt mehr als allein
    assert not A.is_in_core(game, [2 * k, 2 * k])             # keine Aufteilung der Allianzkosten


def test_check_feasible(mini):
    game, k = mini
    assert A.check_feasible(game, [2 * k, 3 * k])
    assert not A.check_feasible(game, [2 * k, 3.5 * k])
    assert not A.check_feasible(game, [np.nan, 5 * k])
    assert not A.check_feasible(game, [5 * k])


def test_member_savings(mini):
    game, k = mini
    s = A.member_savings(game, [2 * k, 3 * k])
    assert s.tolist() == pytest.approx([1 / 3, 1 / 4])


def test_least_core_epsilon_two_members(mini):
    game, k = mini
    eps, x = A.least_core_epsilon(game)
    assert eps == pytest.approx(-k, rel=1e-7)
    assert x.sum() == pytest.approx(5 * k)


def test_least_core_epsilon_detects_an_empty_core(empty_core_game):
    eps, x = A.least_core_epsilon(empty_core_game)
    assert eps == pytest.approx(2 / 3, rel=1e-7)
    assert not A.core_is_nonempty(empty_core_game)
    assert A.core_range(empty_core_game) is None


def test_core_range_two_members_by_hand(mini):
    game, k = mini
    lo, hi = A.core_range(game)
    assert lo.tolist() == pytest.approx([k, 2 * k], rel=1e-6)
    assert hi.tolist() == pytest.approx([3 * k, 4 * k], rel=1e-6)


def test_core_range_is_a_single_point_when_the_core_is_a_point():
    """Zwei Lager, C(1) = 5, C(2) = 5, C(N) = 10: nur x = (5, 5) ist im Kern (kein Gewinn)."""
    game = make_cost_game({(0,): 5, (1,): 5, (0, 1): 10}, 2)
    lo, hi = A.core_range(game)
    assert lo.tolist() == pytest.approx([5.0, 5.0], abs=1e-6)
    assert hi.tolist() == pytest.approx([5.0, 5.0], abs=1e-6)


def test_is_cost_concave_hand_examples(mini):
    game, _ = mini
    assert A.is_cost_concave(game)
    convex = make_cost_game({(0,): 1, (1,): 1, (0, 1): 3}, 2)           # Zusammenlegen verteuert
    assert not A.is_cost_concave(convex)


def test_shapley_trace_averages_to_the_shapley_value(mini):
    game, k = mini
    trace = A.shapley_trace(game, 0)
    assert len(trace) == 2
    assert np.mean([c for _, c in trace]) == pytest.approx(2 * k)
    assert dict(trace)[(0, 1)] == pytest.approx(3 * k) and dict(trace)[(1, 0)] == pytest.approx(k)
