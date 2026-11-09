"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier. Dazu: jedes Preset und die Ränder durchlaufen."""

import numpy as np

import bsa_algorithm as A
import bsa_constants as C
import bsa_evaluation as ev


def _alliance(**over):
    args = dict(gain=0.25, eps_rel=-0.1, core_nonempty=True, shapley_in_core=True, min_saving=0.125, min_member=1)
    args.update(over)
    return ev.verdict_alliance(**args)


def test_de_uses_a_decimal_comma():
    assert ev.de(31.746, 1) == "31,7" and ev.de(0.5, 2) == "0,50" and ev.de(-2.0, 0) == "-2"
    assert ev.de_small(0.0123) == "0,01" and ev.de_small(0.00067) == "0,0007" and ev.de_small(0.01) == "0,01"


def test_alliance_success_names_every_number():
    kind, text = _alliance()
    assert kind == "success"
    assert "spart 25,0 % der Summe der Einzelkosten" in text
    assert "ε* = -10,00 % der Allianzkosten" in text
    assert "Lager 2 mit 12,5 % seiner Einzelkosten" in text


def test_alliance_empty_core_is_a_warning_and_comes_first():
    kind, text = _alliance(core_nonempty=False, eps_rel=0.02, gain=0.001)
    assert kind == "warning" and "Kern ist leer" in text and "ε* = 2,00 %" in text and "0,1 %" in text


def test_alliance_gain_threshold_is_half_a_percent():
    assert _alliance(gain=0.005)[0] == "success"                      # genau 0,5 %: noch ein Gewinn
    kind, text = _alliance(gain=0.0049)
    assert kind == "info" and "0,49 %" in text and "nichts zu verteilen" in text


def test_alliance_without_saving_for_one_member_is_not_a_success():
    kind, text = _alliance(min_saving=0.0, min_member=2)
    assert kind == "info" and "Lager 3 spart" in text and "25,0 %" in text


def test_alliance_shapley_outside_the_core_is_a_warning():
    kind, text = _alliance(shapley_in_core=False, shapley_excess_rel=0.015)
    assert kind == "warning" and "Shapley-Wert liegt nicht im Kern" in text and "25,0 %" in text and "allein 1,50 % der Allianzkosten" in text
    assert "allein 0,0007 % der Allianzkosten" in _alliance(shapley_in_core=False, shapley_excess_rel=6.7e-6)[1]       # winzige Verletzung bleibt sichtbar


def _rows(means):
    """Zeilen für verdict_gain: Raster n in (2, 3) und ρ in (0, 0,5) mit gegebenen Mittelwerten [[n2ρ0, n2ρ.5], [n3ρ0, n3ρ.5]]."""
    return [
        {"n": n, "rho": rho, "mean": means[i][j], "min": means[i][j] - 0.05, "max": means[i][j] + 0.05}
        for i, n in enumerate((2, 3)) for j, rho in enumerate((0.0, 0.5))
    ]


def test_gain_verdict_counts_directions_and_names_extremes():
    kind, text = ev.verdict_gain(_rows([[0.30, 0.10], [0.40, 0.20]]))
    assert kind == "info"
    assert "zwischen 10,0 % (n = 2, ρ = 0,50) und 40,0 % (n = 3, ρ = 0,00)" in text
    assert "kleinste Einzelwert über alle 4 Zellen ist 5,0 %" in text
    assert "sinkt der Mittelwert bei 2 von 2 Lagerzahlen" in text and "steigt er bei 2 von 2 Werten" in text


def test_gain_verdict_counts_an_exception():
    """Steigt der Mittelwert in n bei ρ = 0,5 nicht, steht das in der Zählung (1 von 2)."""
    _, text = ev.verdict_gain(_rows([[0.30, 0.10], [0.40, 0.08]]))
    assert "steigt er bei 1 von 2 Werten" in text


def test_rules_verdict_success_requires_shapley_in_every_instance():
    counts = {C.RULE_EQUAL: 20, C.RULE_MEAN: 45, C.RULE_SINGLE: 98, C.RULE_SHAPLEY: 100}
    kind, text = ev.verdict_rules(counts, 100, "Normal")
    assert kind == "success"
    assert "bei 100 Zufallsinstanzen (Normal)" in text and "Shapley-Wert 100, Proportional zu den Einzelkosten 98, Proportional zum Mittelwert 45, Gleichteilung 20" in text
    assert "kein Beweis" in text and "Reglerrändern" in text
    counts[C.RULE_SHAPLEY] = 99
    assert ev.verdict_rules(counts, 100, "Normal")[0] == "warning"


def test_rules_verdict_with_no_instances_is_not_a_success():
    counts = {rule: 0 for rule in C.ALLOCATION_RULES}
    assert ev.verdict_rules(counts, 0, "Normal")[0] == "warning"


def test_core_empty_verdict():
    kind, text = ev.verdict_core_empty(0, 100, "Lognormal", -0.0125)
    assert kind == "success" and "in 0 von 100 Zufallsinstanzen leer (Lognormal)" in text and "ε* war -1,25 %" in text and "kein Beweis" in text
    kind, text = ev.verdict_core_empty(2, 100, "Normal", 0.01)
    assert kind == "warning" and "in 2 von 100" in text


def _summary(exits, smallest, largest):
    return {"instances": 100, "exits": exits, "smallest": smallest, "largest": largest,
            "by_rho": [{"lo": 0.0, "hi": 0.3, "instances": 30, "exits": 3}, {"lo": 0.3, "hi": 0.6, "instances": 30, "exits": 9}, {"lo": 0.6, "hi": 0.9, "instances": 40, "exits": 20}]}


def test_exit_verdict_names_counts_and_classes():
    kind, text = ev.verdict_exit(_summary(32, 25, 2), C.RULE_EQUAL)
    assert kind == "warning"
    assert "Gleichteilung" in text and "in 32 von 100 Instanzen" in text
    assert "in 25 dieser Fälle die kleinste mittlere Nachfrage, in 2 die größte" in text
    assert "ρ von 0,0 bis 0,3: 3 von 30; ρ von 0,3 bis 0,6: 9 von 30; ρ von 0,6 bis 0,9: 20 von 40" in text


def test_exit_verdict_without_exit_is_a_success():
    kind, text = ev.verdict_exit(_summary(0, 0, 0), C.RULE_SHAPLEY)
    assert kind == "success" and "in keiner der 100 Instanzen" in text


def test_exit_summary_on_handmade_records():
    def rec(rho, mu, exit_values):
        return {"rho": rho, "mu": mu, "single_exit": {C.RULE_EQUAL: np.array(exit_values)}}

    records = [
        rec(0.1, (50.0, 100.0), [0.3, -0.1]),          # Austritt des kleinsten Lagers
        rec(0.4, (100.0, 50.0), [-0.2, 0.05]),         # Austritt des kleinsten Lagers (Index 1)
        rec(0.9, (100.0, 50.0), [0.2, 0.0]),           # Austritt des größten Lagers, obere Klassengrenze inklusive
        rec(0.5, (80.0, 80.0), [-0.1, -0.1]),          # kein Austritt
        rec(0.2, (60.0, 60.0), [0.1, 0.0]),            # Gleichstand der Größe: zählt als kleinstes und als größtes
    ]
    s = ev.exit_summary(records, C.RULE_EQUAL)
    assert (s["instances"], s["exits"], s["smallest"], s["largest"]) == (5, 4, 3, 2)
    assert [(b["instances"], b["exits"]) for b in s["by_rho"]] == [(2, 2), (2, 1), (1, 1)]


def test_every_preset_runs_and_reports_the_expected_verdict_kind():
    """Jedes Preset: Kern nicht leer, Shapley im Kern, Gewinn über der Schwelle, also das Verdikt „success“."""
    for name, p in C.PRESETS.items():
        a = ev.analyse(ev.Settings(**p))
        kind, _ = ev.verdict_alliance(gain=a.gain, eps_rel=a.eps / a.game.grand, core_nonempty=a.core_nonempty,
                                      shapley_in_core=A.is_in_core(a.game, a.rules[C.RULE_SHAPLEY]), min_saving=float(a.savings[C.RULE_SHAPLEY].min()), min_member=0)
        assert kind == "success", name


def test_preset_with_high_correlation_reports_less_gain_than_the_symmetric_one():
    """Die Presets sind nicht wirkungslos: hohe Korrelation senkt den Gewinn gegenüber den drei gleichen Lagern bei gleicher Lagerzahl nicht unter null, aber messbar."""
    base = ev.analyse(ev.Settings(**C.PRESETS["Vier verschieden große Lager"]))
    corr = ev.analyse(ev.Settings(**C.PRESETS["Hohe Korrelation (ρ = 0,8)"]))
    assert corr.gain < base.gain and corr.gain > ev.MIN_GAIN
