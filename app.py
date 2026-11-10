"""Bestandsallianz: wer bekommt den Gewinn des Bündelns? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Mehrere Lager (Spediteure) legen ihre Bestände zusammen: Das Bündeln spart erwartete Kosten, die Aufteilung der Ersparnis entscheidet, ob
jemand austreten will. Gezeigt werden Gewinn, Aufteilungsregeln, Shapley-Wert und der Kern (per LP).

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import bsa_algorithm as A
import bsa_constants as C
import bsa_evaluation as ev
from bsa_layout import dunkles_design, kopf_css
from bsa_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from bsa_texts import ASSUMPTIONS_MARKDOWN
from bsa_visualization import build_allocation, build_exit, build_gain_heatmap, build_rule_counts

st.set_page_config(page_title="Bestandsallianz – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _gain_table(spread, service):
    return ev.gain_table(spread, service)


@st.cache_data(show_spinner=False)
def _series(dist, count):
    return ev.random_records(dist, count)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in bsa_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🤝 Bestandsallianz: wer bekommt den Gewinn des Bündelns?")
st.markdown(
    """
Mehrere Lager (Spediteure) halten je einen Bestand für ihre unsichere Nachfrage. Legen sie die Bestände zusammen, sinken die **erwarteten Kosten** der Allianz unter die Summe der Einzelkosten:
das ist der **Gewinn des Bündelns**. Offen bleibt, **wie** man ihn aufteilt, damit niemand austritt. Eine Aufteilung im **Kern** schützt jede Teilgruppe davor, allein günstiger wegzukommen.

Hier sind alle Stückkosten gleich. Stück der Linie „Gemeinsam gewinnen“; der Vorgänger zeigt, dass das Bündeln Bestand spart, dieses Stück fragt, wie die Ersparnis verteilt wird.
Weiter unten: das **Kernfrage**-Verdikt, vier **🔬 Experimente**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/gewinn-teilen-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Mehrere Lager legen ihre Bestände zusammen und sparen dadurch Kosten. Die Demo fragt, wie man die gemeinsamen Kosten auf die Lager verteilt, damit kein Lager und keine Gruppe lieber allein bliebe. Eine Aufteilung, bei der niemand austreten will, heißt **Kern**. Die Demo rechnet den **Gewinn des Bündelns** und die Aufteilungsregeln gleich teilen, nach der durchschnittlichen Nachfrage, nach den Einzelkosten und nach dem **Shapley-Wert** durch. Dann prüft sie, ob die Aufteilung der jeweiligen Regel im Kern liegt und welches Lager unter welcher Regel austreten würde. **Alle Lager haben dieselben Stückkosten, und die Nachfragen sind erzeugt, nicht gemessen.** Die Zählungen gelten für die gezogenen Zufallsinstanzen und sind kein Beweis.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/gewinn-teilen-erklaert.html)
        """
    )
# erklaerseite:einfach:end

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Lager**")
    n = st.slider("Zahl der Lager n", *bounds("n_slider"), key="n_slider", help="Wie viele Lager die Allianz bilden. Die Zahl der Koalitionen wächst mit 2^n.")
    spread = st.slider(
        "Streuung der Lagergrößen", *bounds("spread_slider"), key="spread_slider", step=C.SPREAD_STEP, format="%.2f",
        help="0: alle Lager gleich groß. Größe = 1 + Streuung · (2u − 1) mit zufälligem u aus [0, 1] je Lager; die Standardabweichung der Normal-Nachfrage je Lager ist 0,3 mal die mittlere Nachfrage mal einen zufälligen Faktor 1 ± 0,5 · Streuung.",
    )
    st.markdown("**Nachfrage**")
    dist = st.selectbox(
        "Verteilung der Nachfrage", C.DISTS, key="dist_select",
        help="Normal: Koalitionskosten aus der Formel. Lognormal und Schwerer Tail (Pareto-Index 2,5): Erwartungswert über 20.000 Zufallsszenarien.",
    )
    rho = st.slider(
        "Gleichkorrelation ρ", *bounds("rho_slider"), key="rho_slider", step=C.RHO_STEP, format="%.2f",
        help="Alle Paare von Lagern gleichlaufend mit ρ (bei Lognormal und Pareto die Korrelation der zugrunde liegenden Normalvariablen). Höchstens 0,95, damit das Bündeln bei Normalverteilung etwas spart (bei Lognormal und Pareto beobachtet, nicht bewiesen).",
    )
    st.markdown("**Kosten**")
    service = st.slider(
        "Servicegrad c_u / (c_u + c_o)", *bounds("service_slider"), key="service_slider", step=C.SERVICE_STEP, format="%.2f",
        help="Verhältnis der Unterbestandskosten zur Summe beider Kostensätze. Kosten stehen in Einheiten von c_o.",
    )
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "rho_slider": float(rho), "spread_slider": float(spread), "service_slider": float(service), "dist_select": dist, "seed_input": int(seed)})

settings = ev.Settings(n=int(n), rho=float(rho), spread=float(spread), service=float(service), dist=dist, seed=int(seed))
a = _analysis(settings)
game = a.game
shapley = a.rules[C.RULE_SHAPLEY]

with st.expander("So funktioniert die Bestandsallianz", expanded=True):
    st.markdown(
        """
1. **Koalitionskosten.** Jede Gruppe S von Lagern bestellt einen gemeinsamen Bestand für die Summe ihrer Nachfragen, so dass der Anteil gedeckter Nachfrage dem Servicegrad entspricht. Die erwarteten Kosten dieses besten Bestands sind C(S).
2. **Gewinn.** Die Allianz aller n Lager kostet C(N), die Lager einzeln zusammen die Summe der C({i}). Der Unterschied ist der Gewinn des Bündelns.
3. **Aufteilung.** Jedes Lager zahlt einen Anteil x_i der Allianzkosten; die Anteile summieren sich zu C(N). Die Regeln unten unterscheiden sich darin, wie sie die Anteile festlegen.
4. **Kern.** Eine Aufteilung liegt im Kern, wenn keine echte Teilgruppe S zusammen weniger als ihren Anteil x(S) = Σ x_i zahlen müsste, falls sie allein bestellt: x(S) ≤ C(S). Ein LP (HiGHS) findet das kleinste ε, für das das für alle Gruppen mit Zuschlag ε geht; der Kern ist genau dann nicht leer, wenn ε* ≤ 0.
        """
    )
    st.markdown("**Die Aufteilungsregeln im Überblick**, mit dem Beitrag von Lager 1 in der aktuellen Instanz als Beispiel (Kosten in Einheiten von c_o):")
    explain = {
        C.RULE_BASELINE: "Jedes Lager bestellt allein und zahlt seine Einzelkosten C({i}). Keine Aufteilung der Allianz, nur der Vergleichswert.",
        C.RULE_EQUAL: "Jedes Lager zahlt C(N) / n, unabhängig von Größe und Risiko.",
        C.RULE_MEAN: "Lager i zahlt C(N) mal seinen Anteil an der gesamten mittleren Nachfrage, μ_i / Σ μ_j.",
        C.RULE_SINGLE: "Lager i zahlt C(N) mal seinen Anteil an den Einzelkosten, C({i}) / Σ C({j}). Bei Normalnachfrage sind die Einzelkosten proportional zu σ_i: dann ist dies zugleich die Regel „proportional zur Standardabweichung“.",
        C.RULE_SHAPLEY: "Lager i zahlt den Mittelwert seiner Zusatzkosten C(S ∪ {i}) − C(S) über alle Reihenfolgen, in denen die Lager der Allianz beitreten.",
    }
    st.dataframe(
        [{"Regel": r, "Rechnung": explain[r], "Beitrag Lager 1": float(a.rules[r][0])} for r in C.RULES],
        hide_index=True, width="stretch",
        column_config={"Beitrag Lager 1": st.column_config.NumberColumn(format="%.2f")},
    )
    if game.n <= 4:
        st.markdown(f"**Beispielspur Shapley-Wert, Lager 1** (n = {game.n}, also {len(A.shapley_trace(game, 0))} Reihenfolgen): Zusatzkosten von Lager 1, wenn die Lager in dieser Reihenfolge beitreten.")
        trace = A.shapley_trace(game, 0)
        st.dataframe(
            [{"Reihenfolge": " → ".join(f"Lager {i + 1}" for i in order), "Zusatzkosten von Lager 1": cost} for order, cost in trace] + [{"Reihenfolge": "Mittel = Shapley-Wert", "Zusatzkosten von Lager 1": float(np.mean([c for _, c in trace]))}],
            hide_index=True, width="stretch", column_config={"Zusatzkosten von Lager 1": st.column_config.NumberColumn(format="%.2f")},
        )
    else:
        st.caption("Die Beispielspur mit allen Reihenfolgen steht bis n = 4 hier; bei größerem n wird der Shapley-Wert über die Teilmengen-Formel gerechnet (Formel unten).")

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
alone = float(game.single.sum())
m1, m2, m3, m4 = st.columns(4)
# Delta immer als „mein Wert minus Referenz“; die Farbe regelt delta_color="inverse" (Vorzeichen-Falle, DEMO-PLAYBOOK §2).
m1.metric("Kosten ohne Allianz", f"{alone:.2f}", delta="Summe der Einzelkosten, Referenz", delta_color="off")
m2.metric("Kosten der Allianz", f"{game.grand:.2f}", delta=f"{game.grand - alone:+.2f} gegen allein", delta_color="inverse")
m3.metric("Gewinn des Bündelns", f"{100 * a.gain:.1f} %".replace(".", ","), delta="in % der Einzelkosten", delta_color="off")
m4.metric("Kern", "nicht leer" if a.core_nonempty else "leer", delta=f"ε* = {ev.de(100 * a.eps / game.grand, 2)} % der Allianzkosten", delta_color="off")
st.plotly_chart(build_allocation(a), width="stretch", key="main_chart")
st.caption(
    "Ersparnis je Lager gegenüber allein (in Prozent seiner Einzelkosten) bei den vier Aufteilungsregeln. Die roten Querstriche zeigen den Kern-Spielraum: "
    "die kleinste und die größte Ersparnis, die ein Lager in irgendeiner Aufteilung im Kern bekommen kann. Bei leerem Kern fehlen sie."
)
st.dataframe(
    ev.rule_rows(a), hide_index=True, width="stretch",
    column_config={
        "Kleinste Ersparnis (%)": st.column_config.NumberColumn(format="%.1f %%"),
        "Größte Ersparnis (%)": st.column_config.NumberColumn(format="%.1f %%"),
        "Größte Unzufriedenheit einer Koalition (% ihrer Kosten)": st.column_config.NumberColumn(format="%.1f %%"),
    },
)
if a.core_lo is not None:
    st.markdown("**Kern-Spielraum je Lager** (Beitrag zu den Allianzkosten über alle Aufteilungen im Kern; Kosten in Einheiten von c_o):")
    st.dataframe(
        ev.core_rows(a), hide_index=True, width="stretch",
        column_config={k: st.column_config.NumberColumn(format="%.2f") for k in ("Einzelkosten", "Kern: kleinster Beitrag", "Kern: größter Beitrag", "Shapley-Wert")},
    )
else:
    st.warning("Der Kern ist leer, einen Kern-Spielraum gibt es nicht.")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Lohnt sich die Allianz für jeden, und gibt es eine Aufteilung ohne Austrittsgrund?")
sav = a.savings[C.RULE_SHAPLEY]
kind, text = ev.verdict_alliance(
    gain=a.gain, eps_rel=a.eps / game.grand, core_nonempty=a.core_nonempty, shapley_in_core=A.is_in_core(game, shapley),
    min_saving=float(sav.min()), min_member=int(np.argmin(sav)), shapley_excess_rel=A.max_excess(game, shapley)[0] / game.grand,
)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Gewinn des Bündelns über Lagerzahl und Korrelation")
st.caption(
    "Normalverteilung, geschlossene Formel; der Gewinn hängt nicht von der Aufteilungsregel ab. Mittel über 20 feste Instanzen je Zelle (andere Lagergrößen je Instanz, Streuung wie in der Seitenleiste). "
    "Der Servicegrad kürzt sich bei der Normalverteilung heraus."
)
rows = _gain_table(settings.spread, settings.service)
st.plotly_chart(build_gain_heatmap(rows), width="stretch", key="gain_chart")
kind, text = ev.verdict_gain(rows)
getattr(st, kind)(text)

st.markdown("---")
st.markdown("### Messreihe für die Experimente 2 bis 4")
st.caption(
    "Die drei folgenden Experimente werten dieselbe Reihe von Zufallsinstanzen aus. Lagerzahl (3 bis 6, bei Szenarien 3 bis 5), ρ (0 bis 0,9), Streuung (0 bis 0,9) und Servicegrad (0,55 bis 0,95) "
    "werden je Instanz aus SplitMix64 gezogen; die Einstellungen der Seitenleiste gelten hier nicht."
)
c1, c2 = st.columns(2)
series_dist = c1.radio("Verteilung der Nachfrage", C.DISTS, key="series_dist", horizontal=True)
series_count = c2.slider("Zahl der Zufallsinstanzen", 20, 200, 100, step=20, key="series_count")
with st.spinner(f"Rechnet {series_count} Zufallsinstanzen ({series_dist}), das dauert einige Sekunden …"):
    records = _series(series_dist, int(series_count))

st.subheader("🔬 Experiment 2: Liegt die Aufteilungsregel im Kern?")
counts = ev.count_in_core(records)
st.plotly_chart(build_rule_counts(counts, len(records)), width="stretch", key="rule_chart")
st.dataframe(
    [
        {
            "Regel": rule, "Im Kern (Instanzen)": counts[rule], "Instanzen gesamt": len(records),
            "Größte Unzufriedenheit einer Koalition (% ihrer Kosten)": 100 * ev.worst_violation(records, rule),
        }
        for rule in C.ALLOCATION_RULES
    ],
    hide_index=True, width="stretch",
    column_config={"Größte Unzufriedenheit einer Koalition (% ihrer Kosten)": st.column_config.NumberColumn(format="%.1f %%")},
)
widths = [r["width_rel"] for r in records if r["width_rel"] is not None]
if widths:
    st.caption(f"Kern-Spielraum: im Mittel über die {len(widths)} Instanzen ist die Spanne der möglichen Beiträge eines Lagers {ev.de(100 * float(np.mean(widths)))} % seiner Einzelkosten breit.")
kind, text = ev.verdict_rules(counts, len(records), series_dist)
getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 3: Ist der Kern leer?")
empty = ev.count_empty_core(records)
worst_eps = max(r["eps_rel"] for r in records)
kind, text = ev.verdict_core_empty(empty, len(records), series_dist, worst_eps)
getattr(st, kind)(text)
st.caption("Voraussetzung: gleiche Stückkosten für alle Lager. Der Kern ist hier das Ergebnis eines LP je Instanz (ε* ≤ 0), nicht eines Beweises.")
if st.button("Alle drei Verteilungen mit dieser Instanzzahl messen", key="three_button"):
    st.session_state["three_count"] = int(series_count)
if "three_count" in st.session_state:
    three_count = st.session_state["three_count"]
    with st.spinner("Rechnet alle drei Verteilungen …"):
        table = []
        for d in C.DISTS:
            recs = _series(d, three_count)
            gains = [r["gain"] for r in recs]
            table.append({
                "Verteilung": d, "Instanzen": len(recs), "Kern leer": ev.count_empty_core(recs), "Größtes ε* (% der Allianzkosten)": 100 * max(r["eps_rel"] for r in recs),
                "Gewinn im Mittel (%)": 100 * float(np.mean(gains)), "Kleinster Gewinn (%)": 100 * float(np.min(gains)),
            })
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={k: st.column_config.NumberColumn(format="%.1f %%") for k in ("Größtes ε* (% der Allianzkosten)", "Gewinn im Mittel (%)", "Kleinster Gewinn (%)")},
    )

st.markdown("---")
st.subheader("🔬 Experiment 4: Wer will austreten, und wann?")
st.caption("Einzelaustritt heißt: Ein Lager zahlt in der Allianz mehr, als es allein zahlen würde (x_i > C({i})). Gewählt wird die Regel; ausgewertet wird dieselbe Reihe wie oben.")
exit_rule = st.selectbox("Aufteilungsregel", C.ALLOCATION_RULES, key="exit_rule")
summary = ev.exit_summary(records, exit_rule)
st.plotly_chart(build_exit(summary), width="stretch", key="exit_chart")
kind, text = ev.verdict_exit(summary, exit_rule)
getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 5: Verweis, nicht nachgebaut")
st.markdown(
    "Den **Nukleolus** (die Aufteilung, die die größte Unzufriedenheit schrittweise minimiert) und die Ersparnis-Aufteilung bei **Touren** zeigt die Demo `kern-demo`. "
    "Den Shapley-Wert für sich, mit Beispielen aus anderen Bereichen, zeigt `shapley-demo`. Hier geht es nur um das Zeitungsjungen-Spiel."
)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Gleiche Stückkosten für alle Lager** | Die Messungen und der zitierte Satz gelten für gleiche Kostensätze. Bei verschiedenen Sätzen ist schon die Definition der Koalitionskosten offen; hier nicht gebaut, nicht gemessen. | - |
| **Keine Umlagerung nach der Realisierung** | Nach Eintreffen der Nachfrage könnten Lager Ware tauschen; das ändert die Kosten und ist ein anderes Spiel. Nicht gebaut. | - |
| **Gleichkorrelation ρ** | Echte Nachfragen hängen nicht paarweise gleich stark zusammen. Die Tests nutzen zusätzlich zufällige Kovarianzmatrizen (`tests/test_claims.py`): der Kern war darin nie leer, der Shapley-Wert lag aber nicht immer im Kern. Die App zeigt nur die Gleichkorrelation. | - |
| **Eine Periode** | Mehrperiodige Allianzen mit Bestandsfortschreibung sind nicht modelliert. | - |
| **Nur Kostenaufteilung, kein Verhandeln** | Der Kern sagt, welche Aufteilungen stabil sind, nicht, welche die Lager verhandeln würden. | `verhandlung-demo`, `kern-demo` |
"""
)

st.markdown("---")

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(ASSUMPTIONS_MARKDOWN)

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/gewinn-teilen-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $n$ Lager mit zufälliger Nachfrage $D_i$ und gleichen Stückkosten. Gesucht sind die Erwartungskosten jeder Gruppe von Lagern, der Gewinn des Bündelns und eine Aufteilung der Allianzkosten, bei der keine Gruppe austreten will.

| Symbol | Bedeutung |
|---|---|
| $N$, $S$ | Menge aller Lager, eine Koalition (Teilmenge) |
| $D_i$, $D_S$ | Nachfrage von Lager $i$, Summe der Nachfragen der Mitglieder von $S$ |
| $c_u$, $c_o$ | Kosten je fehlender bzw. überschüssiger Einheit (gleich für alle) |
| $\alpha$ | Servicegrad $c_u/(c_u+c_o)$ |
| $C(S)$ | Erwartungskosten des besten gemeinsamen Bestands der Koalition $S$ |
| $\sigma_S$, $\Sigma$ | Standardabweichung von $D_S$, Kovarianzmatrix der Nachfragen |
| $z_\alpha$, $\varphi$ | $\alpha$-Quantil der Standardnormalverteilung, deren Dichte |
| $x_i$ | Anteil von Lager $i$ an den Allianzkosten |
| $\varepsilon^*$ | kleinster Zuschlag, mit dem alle Koalitionen zufrieden sind |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/gewinn-teilen-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Koalitionskosten** (bester Bestand $q$ für die Summe der Nachfragen):

$$C(S)=\min_{q\ge 0}\ \mathbb{E}\big[c_u\,(D_S-q)^+ + c_o\,(q-D_S)^+\big],\qquad D_S=\sum_{i\in S}D_i$$

Der beste Bestand ist das $\alpha$-Quantil von $D_S$:

$$q_S^*=F_{D_S}^{-1}(\alpha),\qquad \alpha=\frac{c_u}{c_u+c_o}$$

**Normalfall** (gemeinsam normalverteilt, Kovarianzmatrix $\Sigma$):

$$C(S)=(c_u+c_o)\,\varphi(z_\alpha)\,\sigma_S$$

$$\sigma_S^2=\mathbf{1}_S^{\top}\,\Sigma\,\mathbf{1}_S$$

Bei Gleichkorrelation $\rho$ und Standardabweichungen $\sigma_i$ ist $\sigma_N^2=(1-\rho)\sum_i\sigma_i^2+\rho\big(\sum_i\sigma_i\big)^2$. **Gewinn des Bündelns:**

$$g=1-\frac{C(N)}{\sum_{i\in N}C(\{i\})}$$

Im Normalfall ist $C(N)/\sum_i C(\{i\})=\sigma_N/\sum_i\sigma_i$: der Servicegrad kürzt sich heraus.

**Kern** (Aufteilungen, bei denen keine Koalition austreten will):

$$\text{Kern}=\Big\{x\in\mathbb{R}^n:\ \sum_{i\in N}x_i=C(N),\ \ \sum_{i\in S}x_i\le C(S)\ \text{ für alle } S\subsetneq N\Big\}$$

**Kleinster Zuschlag** (ein LP mit den Variablen $x$ und $\varepsilon$, gelöst mit HiGHS):

$$\varepsilon^*=\min\Big\{\varepsilon:\ \sum_{i\in S}x_i-\varepsilon\le C(S)\ \ \forall\, S\subsetneq N,\ \ \sum_{i\in N}x_i=C(N)\Big\}$$

Der Kern ist genau dann nicht leer, wenn $\varepsilon^*\le 0$. Die Bondareva-Shapley-Bedingung sagt dasselbe über balancierte Familien von Koalitionen: Der Kern ist genau dann nicht leer, wenn für jede Familie nichtnegativer Gewichte $\lambda_S$ mit $\sum_{S\ni i}\lambda_S=1$ für alle $i$ gilt

$$\sum_{S\subsetneq N}\lambda_S\,C(S)\ \ge\ C(N)$$

**Shapley-Wert** (Mittel der Zusatzkosten über alle Reihenfolgen, als Summe über Teilmengen):

$$\phi_i=\sum_{S\subseteq N\setminus\{i\}}\frac{|S|!\,(n-|S|-1)!}{n!}\,\big(C(S\cup\{i\})-C(S)\big)$$

**Kern-Spielraum** von Lager $i$ (je ein LP):

$$\underline{x}_i=\min\{x_i:\ x\in\text{Kern}\},\qquad \overline{x}_i=\max\{x_i:\ x\in\text{Kern}\}$$

**Szenarien.** Bei Lognormal und Pareto ist $C(S)$ der Mittelwert von $c_u(T-q)^+ + c_o(q-T)^+$ über die Szenariosummen $T$ der Koalition mit dem empirischen $\alpha$-Quantil als $q$. Alle Koalitionen verwenden dieselben Szenarien.

**Literatur** (nur, was hier selbst nachgeschlagen wurde):
- A. Müller, M. Scarsini, M. Shaked: *The Newsvendor Game Has a Nonempty Core.* Games and Economic Behavior 38(1), 2002, S. 118–126. Bibliografische Angaben nachgeschlagen (Wikidata); die Aussage, der Kern sei für jede gemeinsame Verteilung der Nachfragen nicht leer, stammt aus der Zusammenfassung in einem Literaturdienst, nicht aus dem Volltext. Die Voraussetzungen (etwa gleiche Kostensätze) wurden nicht im Volltext geprüft.
- B. C. Hartman, M. Dror, M. Shaked: *Cores of Inventory Centralization Games.* Games and Economic Behavior 31, 2000, S. 26–49. Nur die bibliografischen Angaben nachgeschlagen, nicht den Inhalt. Ein Verweis auf Hartman und Dror (2005) ist nicht geprüft und steht hier nicht.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
