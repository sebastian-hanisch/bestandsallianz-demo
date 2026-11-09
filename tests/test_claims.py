"""Jede Zahl der README steht hier (als Band oder exakt) mit ihrer Voraussetzung.

Zufall kommt aus SplitMix64 (`bsa_rng.py`); die Instanzen sind je Reihe fest (`random_settings(index, dist, base)`). Wo eine Zahl vom Löser oder von der Reihenfolge der
Summation abhängen könnte, steht ein Band (Plattform: LP über HiGHS, ndtri und Sortierung über numpy/scipy). Zahlen, die die README zusätzlich aus Scratch-Läufen nennt
(1.000 und 200 Instanzen), stehen dort als „Messung dieser README, nicht im Test“.

Zahl -> Test: Gewinn-Tabelle -> test_gain_table_*; Servicegrad -> test_gain_does_not_depend_on_the_service_level; Kern leer -> test_core_never_empty_*, test_general_correlation_*;
Regeln im Kern -> test_rules_in_core_*; Konkavität -> test_concavity_*; Einzelaustritt -> test_exit_*; Kern-Spielraum -> test_core_width_*; Laufzeit steht nicht im Test."""

from functools import lru_cache

import numpy as np
import pytest
from scipy.special import ndtri

import bsa_algorithm as A
import bsa_constants as C
import bsa_costs as K
import bsa_evaluation as ev
from bsa_rng import uniform_block

BASE_A, BASE_B = C.RANDOM_SEED_BASE, 300000          # zwei Seed-Basen: eine Zahl gilt erst, wenn sie auch mit der zweiten Reihe ungefähr stimmt


def _pct(x: float) -> float:
    return 100.0 * x


@lru_cache(maxsize=None)
def _records(dist: str, count: int, base: int) -> list:
    """Reihe je Prozess einmal rechnen (die Tests lesen sie nur)."""
    return ev.random_records(dist, count, base)


# ---- Experiment 1: Gewinn des Bündelns ----------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gain_rows():
    return ev.gain_table(0.5, 0.75)


def _cell(rows, n, rho):
    return next(r for r in rows if r["n"] == n and r["rho"] == rho)


@pytest.mark.parametrize("n,rho,expected", [
    (2, 0.0, 27.2), (2, 0.5, 12.5), (2, 0.9, 2.4), (4, 0.0, 48.1), (4, 0.5, 20.3), (4, 0.9, 3.7), (8, 0.0, 63.0), (8, 0.5, 24.6), (8, 0.9, 4.4),
])
def test_gain_table_cells(gain_rows, n, rho, expected):
    """Mittel über 20 feste Instanzen je Zelle, Normalverteilung, Streuung 0,5 (Servicegrad ohne Einfluss)."""
    assert _pct(_cell(gain_rows, n, rho)["mean"]) == pytest.approx(expected, abs=0.06)


def test_gain_table_shape_and_smallest_value(gain_rows):
    assert len(gain_rows) == 35 and len(C.SWEEP_SEEDS) == 20
    assert _pct(min(r["min"] for r in gain_rows)) == pytest.approx(2.0, abs=0.06)
    assert min(r["min"] for r in gain_rows) > 0
    assert _pct(min(r["mean"] for r in gain_rows)) == pytest.approx(2.4, abs=0.06) and _pct(max(r["mean"] for r in gain_rows)) == pytest.approx(63.0, abs=0.06)


def test_gain_table_directions_are_counted(gain_rows):
    """Mit wachsendem ρ sinkt der Mittelwert in allen 7 Lagerzahlen, mit wachsender Lagerzahl steigt er bei allen 5 Werten von ρ."""
    _, text = ev.verdict_gain(gain_rows)
    assert "sinkt der Mittelwert bei 7 von 7 Lagerzahlen" in text and "steigt er bei 5 von 5 Werten von ρ" in text


def test_gain_does_not_depend_on_the_service_level():
    """Normalverteilung: der Servicegrad kürzt sich im Gewinn heraus (Servicegrad 0,55 gegen 0,99, sonst gleiche Instanz)."""
    low = ev.analyse(ev.Settings(n=4, rho=0.3, spread=0.5, service=0.55)).gain
    high = ev.analyse(ev.Settings(n=4, rho=0.3, spread=0.5, service=0.99)).gain
    assert low == pytest.approx(high, rel=1e-12)


def test_gain_depends_on_the_service_level_for_scenarios():
    """Bei Szenarien (Lognormal) kürzt sich der Servicegrad nicht heraus: die Gewinne bei 0,55 und 0,99 unterscheiden sich um mehr als einen Prozentpunkt."""
    low = ev.analyse(ev.Settings(n=4, rho=0.3, spread=0.5, service=0.55, dist=C.DIST_LOGNORMAL)).gain
    high = ev.analyse(ev.Settings(n=4, rho=0.3, spread=0.5, service=0.99, dist=C.DIST_LOGNORMAL)).gain
    assert abs(low - high) > 0.01


# ---- Experimente 2 bis 4: Zufallsreihen ---------------------------------------------------------------------------------------------------

SERIES = {
    # (Basis, Verteilung, Zahl): Kern leer, Regeln im Kern (Gleich, Mittelwert, Einzelkosten, Shapley), Gewinn im Mittel und kleinster Gewinn in %
    "A-normal": (BASE_A, C.DIST_NORMAL, 400, (76, 170, 394, 400), (24.5, 3.1)),
    "B-normal": (BASE_B, C.DIST_NORMAL, 400, (93, 183, 397, 400), (23.7, 4.0)),
    "A-lognormal": (BASE_A, C.DIST_LOGNORMAL, 100, (12, 41, 98, 100), (23.7, 4.3)),
    "A-heavy": (BASE_A, C.DIST_HEAVY, 100, (16, 96, 100, 100), (19.9, 3.2)),
}


@pytest.fixture(scope="module", params=list(SERIES))
def series(request):
    base, dist, count, rules, gain = SERIES[request.param]
    return request.param, dist, count, rules, gain, _records(dist, count, base)


def test_core_never_empty_in_the_series(series):
    name, dist, count, _, _, records = series
    assert len(records) == count
    assert ev.count_empty_core(records) == 0, name
    assert max(r["eps_rel"] for r in records) < -1e-4        # ε* deutlich unter 0, nicht knapp an der Toleranz


def test_gain_in_the_series(series):
    _, _, _, _, (mean, smallest), records = series
    gains = [_pct(r["gain"]) for r in records]
    assert np.mean(gains) == pytest.approx(mean, abs=0.3)
    assert min(gains) == pytest.approx(smallest, abs=0.3)


def test_rules_in_core_in_the_series(series):
    name, _, count, (gleich, mittel, einzel, shapley), _, records = series
    counts = ev.count_in_core(records)
    assert counts[C.RULE_SHAPLEY] == shapley == count, name       # Shapley-Wert: in jeder gezogenen Instanz im Kern
    assert counts[C.RULE_SINGLE] == pytest.approx(einzel, abs=2)
    assert counts[C.RULE_MEAN] == pytest.approx(mittel, abs=4)
    assert counts[C.RULE_EQUAL] == pytest.approx(gleich, abs=4)


def test_core_width_in_the_series(series):
    """Kern-Spielraum: Mittel über die Instanzen der mittleren Breite je Lager, in % der Einzelkosten, zwischen 30 und 36,5 in allen vier Reihen (gemessen 30,9 bis 35,7)."""
    widths = [_pct(r["width_rel"]) for r in series[5]]
    assert 30.0 <= float(np.mean(widths)) <= 36.5


def test_exit_counts_in_the_series(series):
    """Einzelaustritt (x_i > C({i})): Shapley und die Regel proportional zu den Einzelkosten nie (mathematisch), Gleichteilung und Mittelwert oft."""
    name, _, count, _, _, records = series
    assert ev.exit_summary(records, C.RULE_SHAPLEY)["exits"] == 0
    assert ev.exit_summary(records, C.RULE_SINGLE)["exits"] == 0
    equal = ev.exit_summary(records, C.RULE_EQUAL)
    assert 0.5 * count < equal["exits"] < 0.8 * count, name
    assert equal["smallest"] >= 0.75 * equal["exits"] and equal["largest"] <= 0.02 * count + 2


def test_exit_numbers_of_the_first_normal_series():
    records = _records(C.DIST_NORMAL, 400, BASE_A)
    equal = ev.exit_summary(records, C.RULE_EQUAL)
    assert equal["exits"] == pytest.approx(245, abs=5) and equal["smallest"] == pytest.approx(206, abs=6) and equal["largest"] == 0
    assert [b["instances"] for b in equal["by_rho"]] == [133, 138, 129]
    assert [b["exits"] for b in equal["by_rho"]] == pytest.approx([55, 85, 105], abs=4)
    mean = ev.exit_summary(records, C.RULE_MEAN)
    assert mean["exits"] == pytest.approx(119, abs=4) and mean["smallest"] == pytest.approx(27, abs=4) and mean["largest"] == pytest.approx(21, abs=4)
    assert [b["exits"] for b in mean["by_rho"]] == pytest.approx([7, 33, 79], abs=4)


def test_exit_of_the_equal_split_rises_with_rho_in_both_normal_series():
    for base in (BASE_A, BASE_B):
        by_rho = ev.exit_summary(_records(C.DIST_NORMAL, 400, base), C.RULE_EQUAL)["by_rho"]
        share = [b["exits"] / b["instances"] for b in by_rho]
        assert share[0] < share[1] < share[2], base


def test_worst_violation_of_the_equal_split_is_large():
    """Gleichteilung, Normal, erste Reihe: die größte Unzufriedenheit einer Koalition liegt über dem Doppelten ihrer eigenen Kosten (468 % in dieser Reihe)."""
    records = _records(C.DIST_NORMAL, 400, BASE_A)
    assert _pct(ev.worst_violation(records, C.RULE_EQUAL)) == pytest.approx(467.7, abs=1.0)
    assert _pct(ev.worst_violation(records, C.RULE_SINGLE)) == pytest.approx(9.8, abs=0.5)
    assert ev.worst_violation(records, C.RULE_SHAPLEY) == 0.0


# ---- Konkavität (warum Shapley im Kern liegt, ist damit nicht erklärt) ---------------------------------------------------------------------


@pytest.mark.parametrize("dist,count,expected", [(C.DIST_NORMAL, 200, 40), (C.DIST_LOGNORMAL, 60, 10), (C.DIST_HEAVY, 60, 31)])
def test_concavity_counts(dist, count, expected):
    """Das Kostenspiel ist nur in einem Teil der Zufallsinstanzen konkav (Grenzkosten fallen mit der Allianzgröße), der Shapley-Wert liegt trotzdem immer im Kern."""
    concave = sum(A.is_cost_concave(ev.build_game(ev.inst_of(ev.random_settings(i, dist)))) for i in range(count))
    assert concave == pytest.approx(expected, abs=3)
    assert concave < count


# ---- Beliebige Korrelationsmatrizen --------------------------------------------------------------------------------------------------------


def _general_cov_game(i: int):
    """Normalspiel mit zufälliger Korrelationsmatrix beliebigen Vorzeichens (normierte A A^T mit A: n x 2n) und Standardabweichungen aus [5, 40]."""
    n = 3 + i % 4
    u = np.clip(uniform_block(700000 + i, n * 2 * n + n + 1), 1e-12, 1 - 1e-12)
    g = ndtri(u[: n * 2 * n]).reshape(n, 2 * n)
    c = g @ g.T
    d = np.sqrt(np.diag(c))
    corr = c / np.outer(d, d)
    sd = 5.0 + 35.0 * u[n * 2 * n : n * 2 * n + n]
    return corr, A.make_game(K.coalition_costs_cov(corr * np.outer(sd, sd), 0.55 + 0.4 * u[-1]))


def test_general_correlation_core_is_never_empty_but_shapley_can_leave_it():
    """200 Normalspiele mit n = 3 bis 6, zufälligen Korrelationen beliebigen Vorzeichens (Gleichkorrelation ist hier nicht vorausgesetzt) und gleichen Stückkosten:
    Der Kern ist in jedem Spiel nicht leer, der Shapley-Wert liegt aber nur in 179 der 200 Spiele im Kern (größte Überschreitung durch eine Koalition 2,4 % der Allianzkosten)."""
    negative, shapley_in, worst = 0, 0, 0.0
    for i in range(200):
        corr, game = _general_cov_game(i)
        negative += bool((corr < -0.05).any())
        eps = A.least_core_epsilon(game)[0]
        assert eps < -1e-4 * game.grand, i
        phi = A.shapley_value(game)
        shapley_in += A.is_in_core(game, phi)
        worst = max(worst, A.max_excess(game, phi)[0] / game.grand)
    assert negative > 150                                     # die Reihe enthält überwiegend echte negative Korrelationen
    assert shapley_in == pytest.approx(179, abs=3) and shapley_in < 200
    assert _pct(worst) == pytest.approx(2.4, abs=0.3)


def test_shapley_leaves_the_core_only_in_a_corner_of_the_slider_range():
    """Raster 7 (n = 2 bis 8) x 6 (ρ = 0, 0,25, 0,5, 0,75, 0,9, 0,95) x 4 (Streuung 0, 0,3, 0,6, 0,9) x 5 Instanzen = 840 Normalspiele, Servicegrad 0,75:
    Der Shapley-Wert liegt in 838 im Kern; die 2 anderen haben n = 8, Streuung 0,9 und ρ >= 0,9, die Überschreitung ist kleiner als 0,001 % der Allianzkosten, der Gewinn kleiner als 5 %."""
    from bsa_scenario import make_instance

    violations = []
    total = 0
    for n in range(2, 9):
        for rho in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95):
            for spread in (0.0, 0.3, 0.6, 0.9):
                for seed in range(5):
                    game = ev.build_game(make_instance(n, rho, spread, 0.75, C.DIST_NORMAL, 1000 + seed))
                    total += 1
                    excess, _ = A.max_excess(game, A.shapley_value(game))
                    if excess > C.TOL * game.grand:
                        violations.append((n, rho, spread, excess / game.grand, A.gain_of_pooling(game)))
    assert total == 840
    assert 1 <= len(violations) <= 4
    assert all(v[0] == 8 and v[2] == 0.9 and v[1] >= 0.9 for v in violations)
    assert max(v[3] for v in violations) < 1e-5
    assert max(v[4] for v in violations) < 0.05                # der Gewinn des Bündelns ist dort kleiner als 5 % der Einzelkosten


def test_single_cost_rule_equals_the_sigma_rule_for_normal_demand():
    """Bei Normalnachfrage sind die Einzelkosten proportional zu sigma_i: die Regel „proportional zu den Einzelkosten“ ist dann dieselbe Aufteilung wie „proportional zur Standardabweichung“,
    und die Regel „proportional zum Mittelwert“ ist davon in den meisten der 20 Instanzen verschieden (die Streuung je Lager weicht vom Mittelwert ab)."""
    different = 0
    for seed in range(20):
        inst = ev.inst_of(ev.random_settings(seed, C.DIST_NORMAL))
        game = ev.build_game(inst)
        single = A.allocate(C.RULE_SINGLE, game, inst.mu)
        assert single == pytest.approx(A.allocation_proportional(game, inst.sd), rel=1e-12)
        different += not np.allclose(A.allocate(C.RULE_MEAN, game, inst.mu), single, rtol=1e-6)
    assert different >= 15
