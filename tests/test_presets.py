"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink (Begrenzen und Einrasten)."""

import pytest

import bsa_constants as C
import bsa_evaluation as ev
import bsa_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_on_the_grid():
    for name, preset in C.PRESETS.items():
        assert C.N_MIN <= preset["n"] <= C.N_MAX, name
        assert C.RHO_MIN <= preset["rho"] <= C.RHO_MAX and C.SPREAD_MIN <= preset["spread"] <= C.SPREAD_MAX, name
        assert C.SERVICE_MIN <= preset["service"] <= C.SERVICE_MAX and preset["dist"] in C.DISTS and 0 <= preset["seed"] <= C.SEED_MAX, name
        for key, spec_key in (("rho", "rho_slider"), ("spread", "spread_slider"), ("service", "service_slider")):
            assert P.snap(preset[key], P.SETTING_SPECS[spec_key]) == pytest.approx(preset[key]), (name, key)


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[C.DEFAULT_PRESET]) == ev.Settings()
    assert next(iter(C.PRESETS)) == C.DEFAULT_PRESET


def test_presets_differ_from_each_other():
    assert len({tuple(sorted(p.items())) for p in C.PRESETS.values()}) == len(C.PRESETS)


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_help_texts_name_the_values_of_their_preset():
    """Die Hilfetexte nennen Zahlen der Presets: sie stehen hier gegen die Preset-Werte."""
    assert C.PRESETS["Hohe Korrelation (ρ = 0,8)"]["rho"] == 0.8
    assert C.PRESETS["Großer Servicegrad"]["service"] == 0.99 and "0,99" in C.PRESET_HELP["Großer Servicegrad"]
    assert C.PRESETS["Vier verschieden große Lager"]["spread"] == 0.8 and "0,8" in C.PRESET_HELP["Vier verschieden große Lager"]
    assert C.PRESETS["Drei gleiche Lager"]["n"] == 3 and C.PRESETS["Drei gleiche Lager"]["rho"] == 0.0
    assert C.PRESETS["Schwerer Tail (Szenarien)"]["dist"] == C.DIST_HEAVY and "2,5" in C.PRESET_HELP["Schwerer Tail (Szenarien)"] and C.PARETO_ALPHA == 2.5
    assert "20.000" in C.PRESET_HELP["Schwerer Tail (Szenarien)"] and C.SCENARIOS == 20000


def test_snap_clamps_and_snaps():
    spec = P.SETTING_SPECS["rho_slider"]
    assert P.snap(5.0, spec) == C.RHO_MAX and P.snap(-1.0, spec) == C.RHO_MIN
    assert P.snap(0.62, spec) == 0.6 and P.snap(0.63, spec) == 0.65
    assert P.snap(100, P.SETTING_SPECS["n_slider"]) == C.N_MAX
    assert P.snap("Normal", P.SETTING_SPECS["dist_select"]) == "Normal" and P.snap("Cauchy", P.SETTING_SPECS["dist_select"]) is None
    assert P.snap(0.757, P.SETTING_SPECS["service_slider"]) == 0.76
