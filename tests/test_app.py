"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Button,
fährt jeden Regler an seine Grenzen, prüft Permalink, Presets, die Verteilungswahl und die Experimente 2 bis 4 und stellt sicher, dass kein Python-Fehler auftritt
(insbesondere keine doppelten Element-IDs und keine Slider mit gleichem Minimum und Maximum)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import bsa_constants as C
from bsa_texts import ASSUMPTIONS_MARKDOWN

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app(query=None) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    for key, value in (query or {}).items():
        at.query_params[key] = value
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at) -> str:
    return " ".join(e.value for kind in (at.success, at.info, at.warning, at.error) for e in kind)


def test_app_loads_without_exception_and_answers_the_core_question():
    at = _fresh_app()
    assert "Ja, im Modell lohnt sich die Allianz für alle" in _texts(at)


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("edge", ["min", "max"])
@pytest.mark.parametrize("dist", C.DISTS)
def test_all_sidebar_controls_at_their_minimum_or_maximum_together(edge, dist):
    at = _fresh_app()
    for key in ("n_slider", "rho_slider", "spread_slider", "service_slider"):
        s = at.sidebar.slider(key=key)
        s.set_value(s.min if edge == "min" else s.max)
    at.sidebar.selectbox(key="dist_select").set_value(dist)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "Kern" in _texts(at)


@pytest.mark.parametrize("dist", C.DISTS)
def test_every_distribution_runs(dist):
    at = _fresh_app()
    at.sidebar.selectbox(key="dist_select").set_value(dist).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_every_preset_runs_and_sets_its_values():
    for name, preset in C.PRESETS.items():
        at = _fresh_app()
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.sidebar.slider(key="n_slider").value == preset["n"]
        assert at.sidebar.slider(key="rho_slider").value == pytest.approx(preset["rho"])
        assert at.sidebar.selectbox(key="dist_select").value == preset["dist"]


def test_permalink_is_clamped_and_snapped():
    at = _fresh_app({"n": "99", "rho": "0.62", "dist": "Lognormal", "spread": "0.3", "service": "0.757", "seed": "42"})
    assert at.sidebar.slider(key="n_slider").value == C.N_MAX
    assert at.sidebar.slider(key="rho_slider").value == pytest.approx(0.6)
    assert at.sidebar.slider(key="service_slider").value == pytest.approx(0.76)
    assert at.sidebar.selectbox(key="dist_select").value == "Lognormal"
    assert at.sidebar.number_input(key="seed_input").value == 42


def test_permalink_ignores_an_unknown_distribution_and_garbage():
    at = _fresh_app({"dist": "Cauchy", "n": "viele"})
    assert at.sidebar.selectbox(key="dist_select").value == C.DEFAULT_DIST
    assert at.sidebar.slider(key="n_slider").value == C.DEFAULT_N


def test_experiment_series_for_every_distribution_and_rule():
    at = _fresh_app()
    for dist in C.DISTS:
        at.radio(key="series_dist").set_value(dist).run(timeout=TIMEOUT)
        assert not at.exception, (dist, [str(e) for e in at.exception])
        assert f"Zufallsinstanzen ({dist})" in _texts(at)
        for rule in C.ALLOCATION_RULES:
            at.selectbox(key="exit_rule").set_value(rule).run(timeout=TIMEOUT)
            assert not at.exception, (dist, rule, [str(e) for e in at.exception])
            assert f"„{rule}“" in _texts(at)


def test_three_distribution_button_shows_a_table():
    at = _fresh_app()
    at.slider(key="series_count").set_value(20).run(timeout=TIMEOUT)
    at.button(key="three_button").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any(set(C.DISTS) <= {str(v) for v in df.value["Verteilung"]} for df in at.dataframe if "Verteilung" in df.value.columns)


def test_expanders_exist_by_title_and_the_assumptions_text_is_shown():
    at = _fresh_app()
    titles = [e.label for e in at.expander]
    for wanted in ("⚠️ Annahmen, Grenzen und bekannte Probleme", "📐 Mathematische Formulierung"):
        assert wanted in titles
    exp = next(e for e in at.expander if e.label == "⚠️ Annahmen, Grenzen und bekannte Probleme")
    assert exp.markdown[0].value.strip() == ASSUMPTIONS_MARKDOWN.strip()


def test_page_has_the_sections_the_intro_announces():
    at = _fresh_app()
    headers = " ".join(h.value for h in at.subheader) + " " + " ".join(m.value for m in at.markdown)
    for part in ("🔬 Experiment 1", "🔬 Experiment 2", "🔬 Experiment 3", "🔬 Experiment 4", "🚧 Wo die Annahmen enden", "❓ Kernfrage"):
        assert part in headers, part
