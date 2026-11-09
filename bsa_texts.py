"""Feste Texte der App, die die README wörtlich wiederholt (ein Test vergleicht beide Fassungen)."""

import bsa_constants as C

SCENARIOS_TEXT = f"{C.SCENARIOS:,}".replace(",", ".")    # 20.000

ASSUMPTIONS_MARKDOWN = f"""
**Annahmen**
- Alle Lager haben **dieselben Stückkosten**: Unterbestandskosten c_u und Überbestandskosten c_o gelten für alle gleich, nur der Servicegrad c_u / (c_u + c_o) ist ein Regler. Kosten stehen in Einheiten von c_o.
- **Eine Periode, ein Bestellzeitpunkt.** Die Allianz bestellt den Bestand für die Summe der Nachfragen und verteilt nur die erwarteten Kosten. Eine Umlagerung nach der Realisierung der Nachfrage gibt es nicht.
- Die Nachfragen sind gemeinsam normalverteilt oder stammen aus Szenarien (Lognormal, Pareto mit schwerem Tail). Die Abhängigkeit ist eine **Gleichkorrelation** ρ (bei den Szenarien eine Gauß-Copula mit Gleichkorrelation); echte Lager sind selten gleich stark verbunden.
- Alle Ersparnisse sind **Erwartungswerte**. Eine einzelne Periode kann anders ausfallen.

**Grenzen**
- Der Kern wird über alle 2^n − 2 echten Koalitionen per LP bestimmt. Der Aufwand wächst mit 2^n, deshalb gibt es höchstens {C.N_MAX} Lager.
- Bei Szenarien sind die Koalitionskosten Schätzungen aus {SCENARIOS_TEXT} Zufallsszenarien (empirischer Erwartungswert) und hängen am Zufalls-Seed.
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
"""
