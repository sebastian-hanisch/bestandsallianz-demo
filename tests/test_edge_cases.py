"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = 1, n = 2, Gleichstände, inklusive Grenzen, Reglerränder."""

import itertools

import numpy as np
import pytest

import bsa_algorithm as A
import bsa_constants as C
import bsa_evaluation as ev
from bsa_scenario import make_instance
from conftest import make_cost_game


def test_exactly_two_members_have_a_core_range_and_a_value():
    """n = 2: der kleinste Fall, in dem es etwas zu verteilen gibt (Zweige `n < 2` greifen nicht)."""
    a = ev.analyse(ev.Settings(n=2))
    assert a.core_lo is not None and len(a.core_lo) == 2
    assert a.in_core[C.RULE_SHAPLEY] and a.gain > 0


def test_single_member_game_is_defined():
    """n = 1: keine echte Koalition; Gewinn 0, Shapley = Einzelkosten, Kern = {C(N)}, kein LP-Fehler."""
    game = make_cost_game({(0,): 7.0}, 1)
    assert A.gain_of_pooling(game) == 0.0
    assert A.shapley_value(game).tolist() == [7.0]
    assert A.least_core_epsilon(game)[0] == 0.0
    lo, hi = A.core_range(game)
    assert lo.tolist() == [7.0] and hi.tolist() == [7.0]
    assert A.is_in_core(game, [7.0])


def test_equal_members_have_equal_shapley_and_equal_rules():
    """Gleichstand: drei gleiche Lager (Streuung 0, symmetrische Korrelation) zahlen unter jeder Regel dasselbe."""
    a = ev.analyse(ev.Settings(n=3, rho=0.4, spread=0.0))
    for rule in C.ALLOCATION_RULES:
        x = a.rules[rule]
        assert x == pytest.approx([x[0]] * 3, rel=1e-9)
    assert all(a.in_core[r] for r in C.ALLOCATION_RULES)         # bei Symmetrie liegen alle vier Regeln im Kern


def test_bounds_of_every_slider_are_inclusive():
    """Jede Reglergrenze (Minimum und Maximum) ist erlaubt und liefert eine zulässige Analyse."""
    lows = dict(n=C.N_MIN, rho=C.RHO_MIN, spread=C.SPREAD_MIN, service=C.SERVICE_MIN)
    highs = dict(n=C.N_MAX, rho=C.RHO_MAX, spread=C.SPREAD_MAX, service=C.SERVICE_MAX)
    for dist in C.DISTS:
        for base in (lows, highs):
            a = ev.analyse(ev.Settings(dist=dist, **base))
            assert A.check_feasible(a.game, a.rules[C.RULE_SHAPLEY])
            assert a.core_nonempty


@pytest.mark.parametrize("name", ["n", "rho", "spread", "service"])
@pytest.mark.parametrize("dist", C.DISTS)
def test_each_slider_alone_at_min_and_max_is_fine(name, dist):
    lows = dict(n=C.N_MIN, rho=C.RHO_MIN, spread=C.SPREAD_MIN, service=C.SERVICE_MIN)
    highs = dict(n=C.N_MAX, rho=C.RHO_MAX, spread=C.SPREAD_MAX, service=C.SERVICE_MAX)
    for edge in (lows, highs):
        a = ev.analyse(ev.Settings(dist=dist, **{name: edge[name]}))
        assert a.gain > 0 and a.core_nonempty


def test_highest_correlation_still_saves_something():
    """ρ = 0,95 (Maximum, damit es einen Gewinn gibt): n = 2, gleiche Lager: Gewinn = 1 - sqrt((1 + ρ) / 2) in der geschlossenen Form."""
    a = ev.analyse(ev.Settings(n=2, rho=0.95, spread=0.0))
    assert a.gain == pytest.approx(1 - np.sqrt((1 + 0.95) / 2), rel=1e-9)
    assert a.gain > 0.01


def test_perfect_correlation_in_the_game_gives_a_point_core():
    """ρ = 1 (nur im Kern-Code, nicht als Regler): kein Gewinn, der Kern ist der Punkt der Einzelkosten, ε* = 0 und der Kern gilt als nicht leer."""
    c = np.array([0.0, 3.0, 4.0, 7.0])
    game = A.make_game(c)
    eps, _ = A.least_core_epsilon(game)
    assert eps == pytest.approx(0.0, abs=1e-9)
    assert A.core_is_nonempty(game)
    lo, hi = A.core_range(game)
    assert lo.tolist() == pytest.approx([3.0, 4.0], abs=1e-6) and hi.tolist() == pytest.approx([3.0, 4.0], abs=1e-6)


def test_all_costs_zero_is_handled():
    game = A.make_game(np.zeros(8))
    assert A.gain_of_pooling(game) == 0.0
    assert A.shapley_value(game).tolist() == [0.0, 0.0, 0.0]
    assert A.least_core_epsilon(game)[0] == 0.0
    assert A.member_savings(game, [0.0, 0.0, 0.0]).tolist() == [0.0, 0.0, 0.0]


def test_service_level_half_has_z_zero():
    a = ev.analyse(ev.Settings(n=3, service=0.5))
    assert a.game.grand > 0 and a.core_nonempty


def test_worst_coalition_excess_is_zero_without_proper_coalition():
    assert ev.worst_coalition_excess_rel(make_cost_game({(0,): 3.0}, 1), [3.0]) == 0.0


def test_ties_in_max_excess_take_the_smallest_mask():
    game = make_cost_game({(0,): 2.0, (1,): 2.0, (0, 1): 3.0}, 2)
    assert A.max_excess(game, [1.5, 1.5])[1] == 1


def test_every_seed_in_a_range_gives_a_nonempty_core_at_the_largest_instance():
    """n = 8, höchstes ρ und höchste Streuung, alle drei Verteilungen: der Kern ist nicht leer (LP mit 254 Nebenbedingungen)."""
    for dist in C.DISTS:
        a = ev.analyse(ev.Settings(n=C.N_MAX, rho=C.RHO_MAX, spread=C.SPREAD_MAX, service=C.SERVICE_MAX, dist=dist, seed=C.SEED_MAX))
        assert a.core_nonempty and a.core_lo is not None


def test_make_instance_n_one():
    inst = make_instance(1, 0.5, 0.5, 0.75, C.DIST_NORMAL, 1)
    assert inst.n == 1
    assert ev.gain_normal(inst) == pytest.approx(0.0, abs=1e-12)


def test_proper_coalition_count():
    """2^n - 2 echte Koalitionen: n = 3 sechs, n = 8 zweihundertvierundfünfzig."""
    assert [2**n - 2 for n in (3, 8)] == [6, 254]
    assert len(list(itertools.chain.from_iterable(itertools.combinations(range(8), r) for r in range(1, 8)))) == 254
