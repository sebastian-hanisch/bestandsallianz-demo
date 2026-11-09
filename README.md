# Bestandsallianz: wer bekommt den Gewinn des Bündelns? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-bestandsallianz-demo.streamlit.app/)** -->

Stück der Linie **Gemeinsam gewinnen** (Arbeitsname; Name, Bereich und Themenfeld werden erst bei der Einbindung in die Website festgelegt). Vorgänger ist [buendelung-demo](https://github.com/sebastian-hanisch/buendelung-demo): Das Bündeln spart Bestand. Dieses Stück fragt, **wie man die Ersparnis aufteilt, damit keiner austritt**. Verwandt: [kern-demo](https://github.com/sebastian-hanisch/kern-demo) (Nukleolus und Aufteilung bei Touren), [shapley-demo](https://github.com/sebastian-hanisch/shapley-demo), [standort-bestand-demo](https://github.com/sebastian-hanisch/standort-bestand-demo), [kuchenteilung-demo](https://github.com/sebastian-hanisch/kuchenteilung-demo) und [verhandlung-demo](https://github.com/sebastian-hanisch/verhandlung-demo).

**Kernfrage:** Lohnt sich eine Bestandsallianz für jeden Teilnehmer, und gibt es immer eine Aufteilung, bei der keine Teilgruppe austreten will (der **Kern**)?

## Warum dieses Problem

Mehrere Spediteure halten je einen Bestand für ihre unsichere Nachfrage. Legen sie die Bestände zusammen, sinken die erwarteten Kosten der Allianz unter die Summe der Einzelkosten. Wer wie viel von den Allianzkosten zahlt, ist eine Aufteilungsfrage: Zahlt eine Teilgruppe in der Allianz mehr, als sie allein zahlen würde, hat sie einen Grund auszutreten. Die Demo rechnet für gleiche Stückkosten den Gewinn des Bündelns, vier Aufteilungsregeln und den Kern, und zählt, wie oft eine Regel im Kern liegt.

## Modell

- **n Lager** (2 bis 8) mit zufälliger Nachfrage `D_i`, gemeinsam **normalverteilt** (Gleichkorrelation ρ von 0 bis 0,95) oder aus **20.000 Szenarien** (Lognormal; schwerer Tail als Pareto mit Index 2,5; Abhängigkeit über eine Gauß-Copula mit Gleichkorrelation). Die Lagergrößen streuen mit dem Regler „Streuung“.
- **Gleiche Stückkosten:** Unterbestandskosten `c_u`, Überbestandskosten `c_o`, Servicegrad `c_u / (c_u + c_o)` von 0,5 bis 0,99; Kosten in Einheiten von `c_o`.
- **Koalitionskosten** `C(S)`: Erwartungskosten des besten gemeinsamen Bestands für die Summe der Nachfragen der Mitglieder. Normal: `(c_u + c_o) · φ(z) · σ_S` mit `σ_S² = 1ᵀ Σ_S 1`; Szenarien: empirischer Erwartungswert mit dem empirischen Quantil als Bestand, alle Koalitionen auf denselben Szenarien.
- **Regeln:** Stand-alone (Vergleichswert, keine Aufteilung), Gleichteilung, proportional zum Mittelwert, proportional zu den Einzelkosten, **Shapley-Wert** (Teilmengen-Formel).
- **Kern** per LP (`scipy.optimize.linprog`, HiGHS): kleinstes ε mit `x(S) ≤ C(S) + ε` für alle echten Koalitionen und `x(N) = C(N)`; der Kern ist genau dann nicht leer, wenn ε* ≤ 0. **Kern-Spielraum** je Mitglied: kleinster und größter Beitrag über alle Aufteilungen im Kern (2n LPs).

Die Formeln stehen im Expander „📐 Mathematische Formulierung“ der App.

## Methodik

- **Kern in kleinen Einheiten** (`bsa_costs.py`: Koalitionskosten; `bsa_algorithm.py`: Gewinn, Shapley, Regeln, Überschuss, Kern-LP, Spielraum; `bsa_evaluation.py`: Kennzahlen, Messreihen, Verdikte als reine Funktionen).
- **Zufall** nur über SplitMix64 (`bsa_rng.py`, wörtliche Kopie aus `ab-test-demo`, Test `tests/test_copies.py`); Szenarien über gleichkorrelierte Standardnormalvariablen und deren Quantiltransformation.
- **Experimente** der App: (1) Gewinn über Lagerzahl und Korrelation (20 feste Instanzen je Zelle), (2) Liegt die Regel im Kern?, (3) Ist der Kern leer?, (4) Wer will austreten, und wann?, (5) Verweis auf `kern-demo`. Die Experimente 2 bis 4 werten eine Reihe von Zufallsinstanzen aus: Lagerzahl 3 bis 6 (bei Szenarien 3 bis 5), ρ von 0 bis 0,9, Streuung von 0 bis 0,9, Servicegrad von 0,55 bis 0,95, aus SplitMix64 gezogen.
- Die Regel „proportional zur Standardabweichung“ aus der Planung ist durch „proportional zum Mittelwert“ ersetzt: Bei Normalnachfrage sind die Einzelkosten proportional zu σ_i, die Regel „proportional zu den Einzelkosten“ ist dann dieselbe Aufteilung (`tests/test_claims.py::test_single_cost_rule_equals_the_sigma_rule_for_normal_demand`).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen gelten für **gleiche Stückkosten** und die Gleichkorrelation des Modells; „Reihe A“ und „Reihe B“ sind Zufallsreihen mit Seed-Basis 200000 und 300000. Prozente von Zählungen sind Zählungen über diese Reihe, keine Wahrscheinlichkeiten. Die Tests prüfen Bänder (Plattform: HiGHS, numpy, scipy); die Zahlen unten stammen vom lokalen Lauf.

**1. Gewinn des Bündelns** (Normalverteilung, Streuung 0,5, Mittel über 20 feste Instanzen je Zelle, 35 Zellen; `tests/test_claims.py::test_gain_table_*`):

| Lagerzahl | ρ = 0 | ρ = 0,5 | ρ = 0,9 |
|---|---|---|---|
| n = 2 | 27,2 % | 12,5 % | 2,4 % |
| n = 4 | 48,1 % | 20,3 % | 3,7 % |
| n = 8 | 63,0 % | 24,6 % | 4,4 % |

Der kleinste Einzelwert über alle Zellen und Instanzen ist 2,0 %, also in jeder gemessenen Zelle größer als 0. Mit wachsendem ρ sinkt der Mittelwert in allen 7 Lagerzahlen, mit wachsender Lagerzahl steigt er bei allen 5 Werten von ρ. Bei Normalverteilung hängt der Gewinn nicht vom Servicegrad ab (Servicegrad 0,55 und 0,99 gleich, `test_gain_does_not_depend_on_the_service_level`); bei Lognormal unterscheiden sich die Gewinne bei diesen Servicegraden um mehr als einen Prozentpunkt (`test_gain_depends_on_the_service_level_for_scenarios`).

**2. Kern leer? In keiner gemessenen Instanz.** (`test_core_never_empty_in_the_series`, `test_general_correlation_core_is_never_empty_but_shapley_can_leave_it`)

| Reihe | Instanzen | Kern leer | Gewinn im Mittel | kleinster Gewinn |
|---|---|---|---|---|
| A, Normal | 400 | 0 | 24,5 % | 3,1 % |
| B, Normal | 400 | 0 | 23,7 % | 4,0 % |
| A, Lognormal | 100 | 0 | 23,7 % | 4,3 % |
| A, schwerer Tail | 100 | 0 | 19,9 % | 3,2 % |
| Normal, zufällige Korrelationsmatrizen beliebigen Vorzeichens, n = 3 bis 6 | 200 | 0 | nicht im Test | nicht im Test |

Das ist eine **Zählung, kein Beweis**. Sie ist mit dem zitierten Satz vereinbar (siehe Literatur; dessen Voraussetzungen sind nicht im Volltext geprüft). Messung dieser README, nicht im Test: Ein Scratch-Lauf mit 1.000 Normal-Instanzen (Seed-Basis 200000) sowie 200 Lognormal- und 200 Tail-Instanzen fand den Kern ebenfalls in keiner Instanz leer.

**3. Liegt die Regel im Kern?** (Zahl der Instanzen mit der Aufteilung im Kern; `test_rules_in_core_in_the_series`):

| Reihe | Gleichteilung | proportional zum Mittelwert | proportional zu den Einzelkosten | Shapley-Wert |
|---|---|---|---|---|
| A, Normal (400) | 76 | 170 | 394 | 400 |
| B, Normal (400) | 93 | 183 | 397 | 400 |
| A, Lognormal (100) | 12 | 41 | 98 | 100 |
| A, schwerer Tail (100) | 16 | 96 | 100 | 100 |

Die Einzelzahlen schwanken zwischen den Reihen (Gleichteilung 76 gegen 93 von 400); die Ordnung der Regeln bleibt gleich. Die größte Überschreitung durch eine Koalition unter der Gleichteilung liegt in Reihe A, Normal, bei 467,7 % der Kosten dieser Koalition, unter „proportional zu den Einzelkosten“ bei 9,8 %. Das sind Extremwerte einer Reihe und schwanken stark: Reihe B (Seed-Basis 300000) kommt auf 600,6 % bzw. 3,2 %, die 9,8 % sind also ein Ausreißer von Reihe A und nicht typisch (`test_worst_violation_of_the_equal_split_is_large`). Der Kern-Spielraum (mittlere Breite der möglichen Beiträge eines Lagers in Prozent seiner Einzelkosten) liegt in den vier Reihen zwischen 30,9 % und 35,7 % (Test: Band von 30 bis 36,5 %, `test_core_width_in_the_series`).

**4. Shapley-Wert und Kern.** In den vier Reihen lag der Shapley-Wert in jeder Instanz im Kern. Das gilt nicht überall (`tests/test_claims.py`):
- In einem Raster von **840 Normalspielen** (n = 2 bis 8, ρ = 0 bis 0,95, Streuung 0 bis 0,9, 5 Instanzen je Zelle) verlässt er den Kern in **2** Spielen: beide mit n = 8, Streuung 0,9 und ρ von mindestens 0,9, um weniger als 0,001 % der Allianzkosten; der Gewinn liegt dort unter 5 % (`test_shapley_leaves_the_core_only_in_a_corner_of_the_slider_range`).
- Bei **zufälligen Korrelationsmatrizen beliebigen Vorzeichens** (200 Spiele, n = 3 bis 6) liegt er in **179** im Kern, in 21 nicht (größte Überschreitung 2,4 % der Allianzkosten); der Kern selbst ist in allen 200 nicht leer.
- Das Kostenspiel (Grenzkosten fallen mit der Allianzgröße) ist nur in einem Teil der Instanzen **konkav**: in 40 von 200 (Normal), 10 von 60 (Lognormal) und 31 von 60 (schwerer Tail; `test_concavity_counts`). Konkavität allein erklärt also nicht, warum der Shapley-Wert in den Reihen im Kern lag; eine Erklärung ist hier nicht gemessen.

**5. Einzelaustritt** (ein Lager zahlt mehr als allein, `x_i > C({i})`; `test_exit_*`): Unter dem Shapley-Wert und unter „proportional zu den Einzelkosten“ gibt es keinen (0 in allen vier Reihen; für „proportional“ folgt `x_i ≤ C({i})` aus `C(N) ≤ Σ C({j})`, für Shapley aus der Subadditivität `C(S ∪ {i}) ≤ C(S) + C({i})` aller Koalitionen bei gleichen Stückkosten, `C(N) ≤ Σ C({j})` allein reicht dafür nicht; die Eigenschaftstests prüfen das an je 15 Instanzen je Verteilung, `tests/test_eigenschaften_bestandsallianz.py`). Unter der Gleichteilung tritt in Reihe A, Normal, in 245 von 400 Instanzen mindestens ein Lager an; das Lager mit dem größten Anreiz hat in 206 dieser Fälle die kleinste mittlere Nachfrage und in keinem die größte. Nach ρ-Klassen (0 bis 0,3 / 0,3 bis 0,6 / 0,6 bis 0,9): 55 von 133, 85 von 138, 105 von 129. Der Anteil steigt mit ρ auch in Reihe B (`test_exit_of_the_equal_split_rises_with_rho_in_both_normal_series`). Unter „proportional zum Mittelwert“ treten in 119 von 400 Instanzen Lager an (7 von 133, 33 von 138, 79 von 129 nach ρ-Klassen); das Lager mit dem größten Anreiz hat in 27 dieser Fälle die kleinste und in 21 die größte mittlere Nachfrage.

**Laufzeit** (Messung dieser README, nicht im Test): eine Analyse bei n = 8, ρ = 0,5, Streuung 0,9, Servicegrad 0,99 dauert lokal 0,04 s (Normal), 0,12 s (Lognormal) und 0,13 s (schwerer Tail), einschließlich des Kern-LP und der 16 LPs des Spielraums. 400 Normal-Instanzen der Reihen brauchen lokal 6,5 bis 8,3 s, 100 Szenario-Instanzen rund 2,5 s.

**Annahmen und Grenzen** (derselbe Text steht in der App im Expander „⚠️ Annahmen, Grenzen und bekannte Probleme“; ein Test vergleicht beide Fassungen):

**Annahmen**
- Alle Lager haben **dieselben Stückkosten**: Unterbestandskosten c_u und Überbestandskosten c_o gelten für alle gleich, nur der Servicegrad c_u / (c_u + c_o) ist ein Regler. Kosten stehen in Einheiten von c_o.
- **Eine Periode, ein Bestellzeitpunkt.** Die Allianz bestellt den Bestand für die Summe der Nachfragen und verteilt nur die erwarteten Kosten. Eine Umlagerung nach der Realisierung der Nachfrage gibt es nicht.
- Die Nachfragen sind gemeinsam normalverteilt oder stammen aus Szenarien (Lognormal, Pareto mit schwerem Tail). Die Abhängigkeit ist eine **Gleichkorrelation** ρ (bei den Szenarien eine Gauß-Copula mit Gleichkorrelation); echte Lager sind selten gleich stark verbunden.
- Alle Ersparnisse sind **Erwartungswerte**. Eine einzelne Periode kann anders ausfallen.

**Grenzen**
- Der Kern wird über alle 2^n − 2 echten Koalitionen per LP bestimmt. Der Aufwand wächst mit 2^n, deshalb gibt es höchstens 8 Lager.
- Bei Szenarien sind die Koalitionskosten Schätzungen aus 20.000 Zufallsszenarien (empirischer Erwartungswert) und hängen am Zufalls-Seed.
- „Kern nicht leer“ heißt: Es gibt eine Aufteilung, bei der keine Teilgruppe austreten will. Es heißt nicht, dass jede Regel eine solche Aufteilung liefert (Experiment 2).
- Den Nukleolus und die Aufteilung bei Touren zeigt die Demo kern-demo; hier werden sie nicht nachgebaut.
- „Allianz“ und „Pool“ sind Modellbegriffe. Die Demo trifft keine rechtliche, steuerliche oder finanzielle Aussage.

**Bekannte Probleme und Unsicherheiten**
- Die Zählungen „Kern leer in k von N Instanzen“ sind Messungen an gezogenen Zufallsinstanzen, **kein Beweis**. Der Satz von Müller, Scarsini und Shaked (2002) wird nur zitiert; seine Voraussetzungen wurden hier nicht im Volltext geprüft.
- Bei zufälligen Korrelationsmatrizen beliebigen Vorzeichens (nur in den Tests, nicht in der App) liegt der Shapley-Wert nicht immer im Kern: in 179 von 200 Spielen tut er es, der Kern selbst war in allen 200 nicht leer (`tests/test_claims.py`). Die Aussage „Shapley im Kern“ der Experimente gilt für die Gleichkorrelation.
- Auch bei Gleichkorrelation liegt der Shapley-Wert nicht überall im Kern: in einem Raster von 840 Normalspielen (n = 2 bis 8, ρ = 0 bis 0,95, Streuung 0 bis 0,9) verließ er ihn in 2 Spielen, beide mit n = 8, Streuung 0,9 und ρ von mindestens 0,9, um weniger als 0,001 % der Allianzkosten (3,2e-6 und 3,4e-6; `tests/test_claims.py`). Dort ist der Gewinn des Bündelns kleiner als 5 % der Einzelkosten, der Kern also schmal. Ob diese zwei Verletzungen echt sind oder numerisches Rauschen der LP- und Summationsrechnung, ist nicht geklärt: ohne Nachrechnung mit höherer Genauigkeit lässt sich Rauschen nicht ausschließen.
- Ungleiche Stückkosten sind nicht gebaut. Was dann mit dem Kern passiert, ist hier weder gerechnet noch gemessen.
- Alle Beispiele sind erzeugt; echte Lagerdaten wurden nicht verwendet, und die Wahl der Verteilungen und Parameter ist willkürlich.
- Die Experimente 2 bis 4 ziehen Lagerzahl, ρ, Streuung und Servicegrad zufällig aus festen Bereichen; andere Bereiche können andere Zählungen ergeben.

## Befunde und Korrekturen gegenüber dem Plan

- **Hook bestätigt:** Der Kern war in keiner gemessenen Instanz leer (siehe Befund 2). Die Plan-Zahlen für den Gewinn des Bündelns (31,7 bis 49,1 % im Mittel, kleinster Wert 11,3 %) wurden **nicht reproduziert**: Mit dem Generator dieser Demo (ρ von 0 bis 0,9 gleichverteilt gezogen, Streuung der Größen) liegt der Gewinn im Mittel bei 19,9 bis 24,5 %, der kleinste Wert bei 3,1 bis 4,3 %. Der Generator der Vormessung war ein anderer (zufällige Korrelationsmatrizen); die Zahlen sind nicht vergleichbar.
- **Neu gegenüber dem Plan:** Der Shapley-Wert liegt nicht immer im Kern (Befund 4). Der Plan nannte das nicht.
- **Literatur des Plans nicht bestätigt:** Der Plan nennt „Hartman und Dror“ für einen leeren Kern bei verschiedenen Kostenparametern. Die Suche bestätigt nur die bibliografischen Angaben von Hartman, Dror und Shaked (2000). Eine Angabe zu späteren Arbeiten (Hartman und Dror 2005, anderes Spiel mit Aufteilung nach Realisierung der Nachfrage) ist nicht geprüft und wird hier nicht als Befund geführt. Nichts davon wurde im Volltext gelesen, nichts davon ist gebaut.
- **Regel „proportional zu σ_i“** ersetzt durch „proportional zum Mittelwert“ (siehe Methodik).
- **Kern-Spielraum bis n = 8** statt bis n = 7: Die 16 LPs bei n = 8 brauchen lokal Bruchteile einer Sekunde (siehe Laufzeit).
- **Kein PDF-Export.**

## Ehrliche Grenzen

Die Annahmen und Grenzen stehen oben im geteilten Textblock. Zusätzlich:
- Alles ist an erzeugten Beispielen gemessen, nicht an Lagerdaten.
- Die Zählungen der Experimente 2 bis 4 gelten für die gezogenen Bereiche (Lagerzahl, ρ, Streuung, Servicegrad) und für die Gleichkorrelation; die Tests prüfen für zufällige Korrelationsmatrizen nur den Kern und den Shapley-Wert, nicht die übrigen Regeln.
- Die Aufteilungsregeln sind eine Auswahl, keine vollständige Liste. Die Gleichteilung und die Regel „proportional zum Mittelwert“ sind einfache Regeln, die oft nicht im Kern liegen; die Regel „proportional zu den Einzelkosten“ lag in den Reihen meist im Kern (Befund 3).
- Dass der Kern nicht leer ist, sagt nichts darüber, welche Aufteilung die Lager verhandeln würden.

**Literatur** (nur, was selbst nachgeschlagen wurde):
- A. Müller, M. Scarsini, M. Shaked: *The Newsvendor Game Has a Nonempty Core.* Games and Economic Behavior 38(1), 2002, S. 118–126 (bibliografische Angaben über Wikidata geprüft). Die Aussage, der Kern sei für jede gemeinsame Verteilung der Nachfragen nicht leer, stammt aus einer Zusammenfassung in einem Literaturdienst, nicht aus dem Volltext; die Voraussetzungen (etwa gleiche Kostensätze) sind nicht geprüft.
- B. C. Hartman, M. Dror, M. Shaked: *Cores of Inventory Centralization Games.* Games and Economic Behavior 31, 2000, S. 26–49. Nur die bibliografischen Angaben nachgeschlagen, nicht den Inhalt. Ein Verweis auf Hartman und Dror (2005) ist nicht geprüft und steht hier nicht.

## Tests

**232 Tests** (`pytest --collect-only -q tests`), lokal in rund 25 s mit `-n 8`. Zusammensetzung:

- **Einheiten gegen Handrechnung:** `test_algorithm.py` (Gewinn, Shapley, Regeln, Überschuss, Kern-LP, Spielraum an Mini-Spielen mit Handwerten, darunter ein Spiel mit leerem Kern, ε* = 2/3), `test_costs.py` (Koalitionskosten, empirischer Bestand), `test_scenario.py` (Instanzen und Szenarien, Bänder).
- **Orakel** (`test_oracle_bestandsallianz.py`, anderer Rechenweg): Koalitionskosten gegen eine Monte-Carlo-Simulation mit Gitter über die Bestellmenge und die Szenario-Engine gegen die Formel; Kern-LP gegen die Aufzählung aller minimal balancierten Familien (n = 3, 90 Spiele, darunter solche mit leerem Kern), gegen das duale LP (n = 3 bis 5) und gegen eine Gittersuche über alle Aufteilungen (n = 3); Shapley gegen das Permutationsmittel und die Teilmengen-Formel (n = 2 bis 5); Kern-Spielraum gegen die geschlossene Form (n = 2) und die Aufzählung aller Ecken des Kerns (n = 3). Das Orakel wird vorher am Handbeispiel geprüft.
- **Eigenschaften** (`test_eigenschaften_bestandsallianz.py`): Umnummerieren, Skalieren, Blindspieler, individuelle Rationalität und Effizienz des Shapley-Werts, Gewinn sinkt mit ρ, Gewinn gleicher Lager in geschlossener Form, gleiche Ausgabe in zwei Prozessen mit verschiedenem `PYTHONHASHSEED`.
- **Claims** (`test_claims.py`): die Zahlen der Befunde 1 bis 5 als Band oder exakt; was nicht im Test steht, ist in der README als „Messung dieser README, nicht im Test“ gekennzeichnet.
- **Randfälle** (`test_edge_cases.py`): n = 1, n = 2, Gleichstände, jeder Regler an Minimum und Maximum für jede Verteilung, ρ = 0,95, alle Kosten null.
- **Verdikte** (`test_verdicts.py`): jede Schwelle und jede Zahl im Text an handgesetzten Werten; jedes Preset durchläuft die Kernfrage.
- **Oberfläche:** `test_app.py` (jeder Button, jeder Regler an Minimum und Maximum, alle Regler zusammen an Minimum und Maximum je Verteilung, jedes Preset, Permalink mit Begrenzen und Einrasten, die Experimente 2 bis 4 für jede Verteilung und Regel), `test_layout.py` (Kopf, Schnellstart in der Seitenleiste, Hervorhebung des aktiven Presets für jeden Regler), `test_presets.py`, `test_footer.py`, `test_runtime_imports.py` (jede Laufzeit-Bibliothek steht in `requirements.txt`, auch scipy), `test_copies.py` (SplitMix64 ist eine wörtliche Kopie), `test_readme.py` (geteilter Text, Testzahl, genannte Testnamen).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Ergebnis, Kernfrage, vier Experimente, Annahmen, Formeln |
| `bsa_algorithm.py` | Kern: Gewinn, Shapley-Wert, Regeln, Überschuss, Kern-LP, Spielraum, Konkavität |
| `bsa_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `bsa_costs.py` | Koalitionskosten: Normalformel und Szenarien |
| `bsa_evaluation.py` | Kennzahlen, Rastertabelle, Zufallsreihen, Zählungen, Verdikte |
| `bsa_layout.py` | Fester Kopf (Stylesheet) |
| `bsa_presets.py` | Permalink (Begrenzen und Einrasten), Presets, Zufalls-Seed-Button |
| `bsa_rng.py` | SplitMix64, wörtliche Kopie von `ab-test-demo/abt_rng.py` |
| `bsa_scenario.py` | Instanzen und Szenarien |
| `bsa_texts.py` | Text „Annahmen, Grenzen und bekannte Probleme“ (App und README) |
| `bsa_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements.txt` | Laufzeit: streamlit, numpy, scipy (HiGHS), plotly |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `tests/` | siehe Abschnitt Tests |

## Bewusst nicht umgesetzt

- **Ungleiche Stückkosten:** Die Koalitionskosten sind dann nicht eindeutig definiert, und ob es dann noch einen Gewinn gibt, ist offen.
- **Umlagerung nach Realisierung der Nachfrage** (Transshipment) und **mehrere Perioden.**
- **Nukleolus:** Verweis auf `kern-demo`.
- **PDF-Export** und eine eigene Erklärseite (die Website-Einbindung kommt später).
- **Beliebige Korrelationsmatrizen in der App** (nur in den Tests).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die CI läuft mit Python 3.12; die Python-Version der Streamlit Cloud wird erst beim Deploy festgelegt.

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
