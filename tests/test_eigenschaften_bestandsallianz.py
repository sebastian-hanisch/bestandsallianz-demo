"""Eigenschaftstests (metamorph, ORACLE-PLAYBOOK §5.2, Zeile „Spieltheorie“): Aussagen, die ohne Kenntnis des richtigen Ergebnisses gelten müssen."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import bsa_algorithm as A
import bsa_constants as C
import bsa_evaluation as ev
from bsa_rng import uniform_block
from bsa_scenario import make_instance

ROOT = Path(__file__).resolve().parent.parent


def _game(seed: int, n: int = 4, dist: str = C.DIST_NORMAL) -> A.Game:
    u = uniform_block(31000 + seed, 3)
    return ev.build_game(make_instance(n, float(0.9 * u[0]), float(0.9 * u[1]), float(0.55 + 0.4 * u[2]), dist, 5000 + seed))


def _permute(game: A.Game, perm) -> A.Game:
    """Spiel mit umnummerierten Mitgliedern: neues Mitglied j ist altes Mitglied perm[j]."""
    n = game.n
    cost = np.zeros(2**n)
    for s in range(2**n):
        old = sum(1 << perm[j] for j in range(n) if s >> j & 1)
        cost[s] = game.cost[old]
    return A.make_game(cost)


@pytest.mark.parametrize("seed", range(12))
def test_renaming_the_members_only_renames_the_values(seed):
    game = _game(seed)
    perm = [2, 0, 3, 1]
    renamed = _permute(game, perm)
    assert A.shapley_value(renamed) == pytest.approx(A.shapley_value(game)[perm], rel=1e-9)
    assert A.least_core_epsilon(renamed)[0] == pytest.approx(A.least_core_epsilon(game)[0], abs=1e-7 * game.grand)
    lo, hi = A.core_range(game)
    lo2, hi2 = A.core_range(renamed)
    assert lo2 == pytest.approx(lo[perm], abs=1e-6 * game.grand) and hi2 == pytest.approx(hi[perm], abs=1e-6 * game.grand)


@pytest.mark.parametrize("seed", range(8))
def test_scaling_all_costs_scales_every_value(seed):
    game = _game(seed, n=3)
    scaled = A.make_game(game.cost * 3.5)
    assert A.shapley_value(scaled) == pytest.approx(3.5 * A.shapley_value(game), rel=1e-9)
    assert A.least_core_epsilon(scaled)[0] == pytest.approx(3.5 * A.least_core_epsilon(game)[0], rel=1e-6)
    assert A.gain_of_pooling(scaled) == pytest.approx(A.gain_of_pooling(game), rel=1e-12)
    assert A.core_range(scaled)[0] == pytest.approx(3.5 * A.core_range(game)[0], rel=1e-5, abs=1e-6 * game.grand)


def test_a_dummy_member_pays_nothing_and_changes_nobody_else():
    """Ein Mitglied, das nie etwas kostet (C(S + d) = C(S)), zahlt im Shapley-Wert null, die anderen zahlen wie ohne es."""
    base = _game(3, n=3)
    cost = np.zeros(16)
    for s in range(8):
        cost[s] = base.cost[s]
        cost[s | 8] = base.cost[s]                         # Mitglied 4 (Bit 3) ist ein Blindspieler
    with_dummy = A.make_game(cost)
    phi = A.shapley_value(with_dummy)
    assert phi[3] == pytest.approx(0.0, abs=1e-9)
    assert phi[:3] == pytest.approx(A.shapley_value(base), rel=1e-9)


@pytest.mark.parametrize("dist", C.DISTS)
def test_shapley_is_individually_rational_and_efficient_in_every_instance(dist):
    """Subadditivität (Bündeln kostet nie mehr als getrennt) folgt aus der Rechnung; daraus x_i <= C({i}) für den Shapley-Wert und die Regel proportional zu den Einzelkosten."""
    for seed in range(15):
        game = _game(seed, n=4, dist=dist)
        phi = A.shapley_value(game)
        assert phi.sum() == pytest.approx(game.grand, rel=1e-9)
        assert np.all(phi <= game.single * (1 + 1e-9))
        prop = A.allocate(C.RULE_SINGLE, game, [1.0] * 4)
        assert np.all(prop <= game.single * (1 + 1e-9))


def test_gain_does_not_increase_with_the_correlation():
    for seed in range(20):
        gains = [ev.gain_normal(make_instance(5, rho, 0.6, 0.75, C.DIST_NORMAL, seed)) for rho in (0.0, 0.25, 0.5, 0.75, 0.95)]
        assert all(a > b for a, b in zip(gains, gains[1:])), seed


def test_equal_correlation_gain_for_equal_members_is_the_closed_form():
    """n gleiche Lager, Gleichkorrelation ρ: Gewinn = 1 - sqrt((1 + (n - 1) ρ) / n)."""
    for n in (2, 3, 5, 8):
        for rho in (0.0, 0.4, 0.9):
            inst = make_instance(n, rho, 0.0, 0.75, C.DIST_NORMAL, 1)
            assert ev.gain_normal(inst) == pytest.approx(1 - np.sqrt((1 + (n - 1) * rho) / n), rel=1e-9)


def test_same_settings_give_the_same_analysis_in_another_process():
    """Dasselbe Szenario in zwei Prozessen mit verschiedenem PYTHONHASHSEED: gleiche Kernkennzahlen (keine Abhängigkeit von Mengenreihenfolgen)."""
    code = (
        "import json, sys; sys.path.insert(0, r'%s'); import bsa_evaluation as ev, bsa_constants as C;"
        "a = ev.analyse(ev.Settings(n=5, rho=0.3, spread=0.6, service=0.8, dist=C.DIST_LOGNORMAL, seed=7));"
        "print(json.dumps([round(a.gain, 12), round(a.eps, 9), [round(float(v), 9) for v in a.rules[C.RULE_SHAPLEY]]]))" % ROOT
    )
    outputs = []
    for hashseed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True, timeout=120).stdout
        outputs.append(json.loads(out.strip().splitlines()[-1]))
    flat = [[o[0], o[1], *o[2]] for o in outputs]
    assert flat[0] == pytest.approx(flat[1], rel=1e-9)
