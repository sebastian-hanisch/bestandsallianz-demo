"""Orakel-Tests der Bestandsallianz: jeder Kern gegen einen anderen Rechenweg (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §2 Zeile „Spieltheorie“).

1. Koalitionskosten C(S): Monte-Carlo-Zeitungsjunge (Cholesky-Simulation, Bestellmenge per Gitter) gegen die Normalformel; Szenario-Engine gegen die Formel.
2. Kern per LP gegen die Bondareva-Shapley-Seite: bei n = 3 gegen die Aufzählung aller minimal balancierten Familien (exakt, ohne LP), bei n = 3 bis 5 gegen das duale LP,
   bei n = 3 gegen eine Gittersuche über alle Aufteilungen.
3. Shapley-Wert: gegen das Mittel über alle Permutationen und gegen die Teilmengen-Formel in reinem Python.
4. Kern-Spielraum: bei n = 2 gegen die geschlossene Form, bei n = 3 gegen die Aufzählung aller Ecken des Kerns (lineare Gleichungssysteme, ohne LP).
Alle Instanzen kommen aus SplitMix64 (feste Seeds); Laufzeit zusammen unter 10 Sekunden."""

import itertools
import math

import numpy as np
import pytest

import bsa_algorithm as A
import bsa_constants as C
import bsa_costs as K
import bsa_evaluation as ev
from bsa_rng import uniform_block
from bsa_scenario import covariance, make_instance
from conftest import make_cost_game

# ---- Hilfen: zufällige Spiele und Referenz-Rechnungen --------------------------------------------------------------------------------------


def random_cost_game(n: int, seed: int) -> A.Game:
    """Zufälliges Kostenspiel ohne weitere Struktur (der Kern kann leer sein): C(S) = Zufallswert aus [1, 10] mal |S|^0,8, je Koalition unabhängig gezogen."""
    u = uniform_block(seed, 2**n)
    sizes = np.array([bin(s).count("1") for s in range(2**n)], dtype=float)
    cost = (1.0 + 9.0 * u) * sizes**0.8
    cost[0] = 0.0
    return A.make_game(cost)


def epsilon_by_balanced_families_n3(game: A.Game) -> float:
    """Exakt bei n = 3: epsilon* = max über die minimal balancierten Familien B von (C(N) - sum_S w_S C(S)) / (sum_S w_S) (Gewichte w mit Überdeckung 1 je Lager)."""
    c = game.cost
    families = [
        ({1: 1.0, 2: 1.0, 4: 1.0}),                       # {1}, {2}, {3}
        ({3: 1.0, 4: 1.0}), ({5: 1.0, 2: 1.0}), ({6: 1.0, 1: 1.0}),     # {12}{3}, {13}{2}, {23}{1}
        ({3: 0.5, 5: 0.5, 6: 0.5}),                       # {12}, {13}, {23} mit Gewicht 1/2
    ]
    return max((game.grand - sum(w * c[s] for s, w in fam.items())) / sum(fam.values()) for fam in families)


def epsilon_by_dual_lp(game: A.Game) -> float:
    """Duales LP des Kern-LP: max mu C(N) - sum lambda_S C(S) mit sum lambda = 1, sum_{S enthält i} lambda_S = mu für alle i, lambda >= 0 (über echte Koalitionen)."""
    from scipy.optimize import linprog

    n = game.n
    masks = K.mask_matrix(n)[1 : 2**n - 1]
    m = len(masks)
    cost = game.cost[1 : 2**n - 1]
    obj = np.append(cost, -game.grand)                           # minimiere sum lambda C - mu C(N)
    a_eq = np.vstack([np.append(np.ones(m), 0.0), np.hstack([masks.T, -np.ones((n, 1))])])
    b_eq = np.append(1.0, np.zeros(n))
    res = linprog(obj, A_eq=a_eq, b_eq=b_eq, bounds=[(0, None)] * m + [(None, None)], method="highs")
    assert res.status == 0
    return -float(res.fun)


def shapley_by_permutations(game: A.Game) -> np.ndarray:
    """Mittel der Zusatzkosten über alle n! Beitrittsreihenfolgen."""
    n = game.n
    total = np.zeros(n)
    for order in itertools.permutations(range(n)):
        joined = 0
        for i in order:
            total[i] += game.cost[joined | (1 << i)] - game.cost[joined]
            joined |= 1 << i
    return total / math.factorial(n)


def shapley_by_subsets(game: A.Game) -> np.ndarray:
    """Teilmengen-Formel in reinem Python über Python-Mengen."""
    n = game.n
    players = list(range(n))
    phi = np.zeros(n)
    for i in players:
        others = [j for j in players if j != i]
        for r in range(n):
            for sub in itertools.combinations(others, r):
                s = sum(1 << j for j in sub)
                weight = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                phi[i] += weight * (game.cost[s | (1 << i)] - game.cost[s])
    return phi


def core_vertices_n3(game: A.Game) -> np.ndarray:
    """Alle Ecken des Kerns bei n = 3: je zwei echte Koalitionen mit x(S) = C(S) und x(N) = C(N) als lineares System, danach die Zulässigkeit prüfen."""
    masks = K.mask_matrix(3)[1:7]
    cost = game.cost[1:7]
    out = []
    for i, j in itertools.combinations(range(6), 2):
        mat = np.vstack([masks[i], masks[j], np.ones(3)])
        if abs(np.linalg.det(mat)) < 1e-9:
            continue
        x = np.linalg.solve(mat, [cost[i], cost[j], game.grand])
        if np.all(masks @ x <= cost + 1e-9 * game.grand):
            out.append(x)
    return np.array(out)


def normal_games(n: int, seeds, rho_range=(0.0, 0.9)):
    for s in seeds:
        u = uniform_block(900000 + s, 3)
        rho = rho_range[0] + u[0] * (rho_range[1] - rho_range[0])
        yield ev.build_game(make_instance(n, float(rho), float(0.9 * u[1]), float(0.55 + 0.4 * u[2]), C.DIST_NORMAL, 4000 + s))


# ---- 1. Koalitionskosten ------------------------------------------------------------------------------------------------------------------


def test_normal_formula_against_monte_carlo_newsvendor():
    """Cholesky-Simulation gemeinsam normaler Nachfragen, Bestellmenge als Minimum über ein feines Gitter: Abweichung von der Formel unter 2 % für jede Koalition."""
    inst = make_instance(3, 0.5, 0.6, 0.8, C.DIST_NORMAL, 17)
    cov = covariance(inst)
    m = 120_000
    u = np.clip(uniform_block(31, m * 3), 1e-12, 1 - 1e-12)
    from scipy.special import ndtri

    z = ndtri(u).reshape(m, 3)
    d = np.array(inst.mu) + z @ np.linalg.cholesky(cov).T
    cu = 0.8 / 0.2
    formula = K.coalition_costs_normal(inst)
    for s in range(1, 8):
        t = d[:, [i for i in range(3) if s >> i & 1]].sum(axis=1)
        grid = np.linspace(t.mean() - 4 * t.std(), t.mean() + 4 * t.std(), 161)
        best = min(float(np.mean(cu * np.maximum(t - q, 0) + np.maximum(q - t, 0))) for q in grid)
        assert best == pytest.approx(formula[s], rel=0.02), s


def test_scenario_engine_matches_the_normal_formula_for_normal_demands():
    """Dieselbe Normalnachfrage einmal über die Formel, einmal über 20.000 Szenarien (Faktormodell, empirisches Quantil): 3 % Band je Koalition."""
    inst = make_instance(3, 0.4, 0.5, 0.75, C.DIST_NORMAL, 9)
    from bsa_scenario import scenario_demands

    engine = K.coalition_costs_scenarios(scenario_demands(inst), inst.service)
    formula = K.coalition_costs_normal(inst)
    assert engine[1:] == pytest.approx(formula[1:], rel=0.03)


def test_gain_from_the_formula_matches_the_game():
    for seed in range(10):
        inst = make_instance(5, 0.3, 0.7, 0.8, C.DIST_NORMAL, seed)
        assert ev.gain_normal(inst) == pytest.approx(A.gain_of_pooling(ev.build_game(inst)), rel=1e-9)


# ---- 2. Kern ------------------------------------------------------------------------------------------------------------------------------


def test_balanced_family_enumeration_on_the_hand_example(empty_core_game):
    """Das Handbeispiel (2/3) prüft das Orakel selbst, bevor es etwas anderes prüft."""
    assert epsilon_by_balanced_families_n3(empty_core_game) == pytest.approx(2 / 3)
    assert epsilon_by_dual_lp(empty_core_game) == pytest.approx(2 / 3, rel=1e-7)


def test_least_core_matches_balanced_family_enumeration_n3():
    games = [random_cost_game(3, s) for s in range(60)] + list(normal_games(3, range(30)))
    empty = 0
    for game in games:
        eps = A.least_core_epsilon(game)[0]
        assert eps == pytest.approx(epsilon_by_balanced_families_n3(game), abs=1e-7 * game.grand)
        empty += eps > 1e-9 * game.grand
    assert 0 < empty < len(games)                      # die Zufallsspiele enthalten leere und nichtleere Kerne: das Orakel unterscheidet beide


@pytest.mark.parametrize("n", [3, 4, 5])
def test_least_core_matches_the_dual_lp(n):
    games = [random_cost_game(n, 100 * n + s) for s in range(25)] + list(normal_games(n, range(10)))
    for game in games:
        assert A.least_core_epsilon(game)[0] == pytest.approx(epsilon_by_dual_lp(game), abs=1e-7 * game.grand)


def test_least_core_matches_a_grid_search_over_all_allocations_n3():
    """Gitter über (x_1, x_2), x_3 = C(N) - x_1 - x_2: kleinstes Maximum der Überschüsse. Das LP darf nicht schlechter sein als jeder Gitterpunkt, und das Gitter liegt höchstens eine Gitterweite darüber."""
    masks = K.mask_matrix(3)[1:7]
    for game in list(normal_games(3, range(8))) + [random_cost_game(3, s) for s in range(8)]:
        h = game.grand / 100.0
        axis = np.arange(-0.3 * game.grand, 1.3 * game.grand, h)
        x1, x2 = np.meshgrid(axis, axis)
        x = np.stack([x1.ravel(), x2.ravel(), game.grand - x1.ravel() - x2.ravel()], axis=1)
        worst = (x @ masks.T - game.cost[1:7]).max(axis=1)
        eps = A.least_core_epsilon(game)[0]
        assert worst.min() >= eps - 1e-9 * game.grand
        assert worst.min() <= eps + 1.01 * h


def test_the_lp_allocation_attains_epsilon():
    """Die zurückgegebene Aufteilung hat genau den größten Überschuss epsilon* (ein Zertifikat, das sich ohne das LP nachrechnen lässt)."""
    for game in list(normal_games(4, range(10))) + [random_cost_game(4, s) for s in range(10)]:
        eps, x = A.least_core_epsilon(game)
        assert A.check_feasible(game, x)
        assert A.max_excess(game, x)[0] == pytest.approx(eps, abs=1e-7 * game.grand)


# ---- 3. Shapley ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_shapley_matches_permutations_and_subset_formula(n):
    games = [random_cost_game(n, 7 * n + s) for s in range(10)] + list(normal_games(n, range(5)))
    for game in games:
        phi = A.shapley_value(game)
        assert phi == pytest.approx(shapley_by_permutations(game), rel=1e-9, abs=1e-9)
        assert phi == pytest.approx(shapley_by_subsets(game), rel=1e-9, abs=1e-9)
        assert phi.sum() == pytest.approx(game.grand, rel=1e-9)


def test_shapley_on_a_scenario_game():
    game = ev.build_game(make_instance(4, 0.3, 0.5, 0.75, C.DIST_LOGNORMAL, 3))
    assert A.shapley_value(game) == pytest.approx(shapley_by_permutations(game), rel=1e-9)


# ---- 4. Kern-Spielraum --------------------------------------------------------------------------------------------------------------------


def test_core_range_two_members_closed_form():
    for game in normal_games(2, range(20)):
        lo, hi = A.core_range(game)
        c1, c2 = game.single
        assert lo.tolist() == pytest.approx([game.grand - c2, game.grand - c1], rel=1e-6)
        assert hi.tolist() == pytest.approx([c1, c2], rel=1e-6)


def test_core_range_three_members_against_vertex_enumeration():
    checked = 0
    for game in list(normal_games(3, range(30))) + [ev.build_game(make_instance(3, 0.3, 0.5, 0.75, d, 12)) for d in (C.DIST_LOGNORMAL, C.DIST_HEAVY)]:
        vertices = core_vertices_n3(game)
        assert len(vertices) >= 1
        lo, hi = A.core_range(game)
        assert lo == pytest.approx(vertices.min(axis=0), rel=1e-6, abs=1e-6 * game.grand)
        assert hi == pytest.approx(vertices.max(axis=0), rel=1e-6, abs=1e-6 * game.grand)
        checked += 1
    assert checked == 32


def test_random_core_points_lie_inside_the_range_and_in_the_core():
    """Konvexkombinationen der Ecken (Zufallsgewichte aus SplitMix64) sind Kernpunkte: im Kern und innerhalb des Spielraums."""
    for game in normal_games(3, range(10)):
        vertices = core_vertices_n3(game)
        lo, hi = A.core_range(game)
        w = uniform_block(5, 40 * len(vertices)).reshape(40, len(vertices))
        w = w / w.sum(axis=1, keepdims=True)
        for x in w @ vertices:
            assert A.is_in_core(game, x, tol=1e-7)
            assert np.all(x >= lo - 1e-6 * game.grand) and np.all(x <= hi + 1e-6 * game.grand)


def test_empty_core_has_no_range_and_vertex_enumeration_finds_no_vertex(empty_core_game):
    assert A.core_range(empty_core_game) is None
    assert len(core_vertices_n3(empty_core_game)) == 0


def test_make_cost_game_helper_roundtrip():
    game = make_cost_game({(0,): 1.0, (1,): 2.0, (0, 1): 2.5}, 2)
    assert game.cost.tolist() == [0.0, 1.0, 2.0, 2.5]
