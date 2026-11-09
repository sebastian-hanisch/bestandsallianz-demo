"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Kosten stehen in Einheiten des Überbestandskostensatzes c_o (c_o = 1); der Unterbestandskostensatz folgt aus dem Servicegrad:
c_u / (c_u + c_o) = Servicegrad, also c_u = Servicegrad / (1 - Servicegrad)."""

N_MIN, N_MAX, DEFAULT_N = 2, 8, 3          # n Lager; 2^n Koalitionen: bei n = 8 sind es 256 (Komplexität Kern-LP: 2^n Nebenbedingungen)
RHO_MIN, RHO_MAX, DEFAULT_RHO, RHO_STEP = 0.0, 0.95, 0.0, 0.05   # Gleichkorrelation der Nachfragen; unter 1, damit das Bündeln bei Normalverteilung etwas spart
SPREAD_MIN, SPREAD_MAX, DEFAULT_SPREAD, SPREAD_STEP = 0.0, 0.9, 0.0, 0.05   # Streuung der Lagergrößen: Größe = 1 + Streuung * (2u - 1), u gleichverteilt
SERVICE_MIN, SERVICE_MAX, DEFAULT_SERVICE, SERVICE_STEP = 0.50, 0.99, 0.75, 0.01
DIST_NORMAL, DIST_LOGNORMAL, DIST_HEAVY = "Normal", "Lognormal", "Schwerer Tail"
DISTS = (DIST_NORMAL, DIST_LOGNORMAL, DIST_HEAVY)
DEFAULT_DIST = DIST_NORMAL
SEED_MAX, DEFAULT_SEED = 999999, 35

CO = 1.0                    # Überbestandskostensatz, Recheneinheit aller Kosten
MU_BASE = 100.0             # mittlere Nachfrage eines Lagers der Größe 1
CV_NORMAL = 0.3             # Variationskoeffizient der Normal-Nachfrage bei Größe 1 (Streuung sigma_i = CV * mu_i * Faktor)
LN_SHAPE = 0.5              # Lognormal: D = mu * exp(LN_SHAPE * Z - LN_SHAPE^2 / 2), Mittel mu
PARETO_ALPHA = 2.5          # schwerer Tail: Pareto mit Index 2,5 (endliche Varianz), Mittel mu, Abhängigkeit über eine Gauß-Copula
SCENARIOS = 20000           # Szenarien je Instanz; Komplexität: O(m * 2^n * log m) bei n = 8 rund 256 Sortierungen von 20.000 Werten
SCEN_SEED_OFFSET = 10**9    # der Szenario-Strom nutzt Seed + Offset, getrennt vom Strom der Lagergrößen
TOL = 1e-9                  # relative Toleranz der Kern-Prüfung (Anteil der Gesamtkosten)

N_SWEEP = (2, 3, 4, 5, 6, 7, 8)
RHO_SWEEP = (0.0, 0.25, 0.5, 0.75, 0.9)
SWEEP_SEEDS = tuple(range(100000, 100020))   # 20 feste Instanzen je Zelle von Experiment 1: nie ein einzelner Seed als Befund
RANDOM_N_MIN, RANDOM_N_MAX = 3, 6            # Lagerzahl der Zufallsinstanzen in den Experimenten 2 bis 4
RANDOM_SEED_BASE = 200000
COUNT_NORMAL, COUNT_SCENARIO = 200, 30        # Zahl der Zufallsinstanzen live (Normal, Szenarien)
COUNT_MAX_NORMAL, COUNT_MAX_SCENARIO = 400, 60
RANDOM_N_MAX_SCENARIO = 5                    # Szenario-Instanzen: bis n = 5 (32 Koalitionen), damit der Lauf kurz bleibt

RULES = ("Stand-alone (ohne Allianz)", "Gleichteilung", "Proportional zum Mittelwert", "Proportional zu den Einzelkosten", "Shapley-Wert")
RULE_BASELINE, RULE_EQUAL, RULE_MEAN, RULE_SINGLE, RULE_SHAPLEY = RULES
ALLOCATION_RULES = RULES[1:]                 # Stand-alone ist keine Aufteilung (summiert sich nicht zu den Allianzkosten)

COLOR_RULES = {
    RULE_EQUAL: "#9e9e9e", RULE_MEAN: "#e6a23c", RULE_SINGLE: "#4c78a8", RULE_SHAPLEY: "#2F6B65",
}
COLOR_CORE = "#b03a2e"
COLOR_LINE = "#4c78a8"

PRESETS = {
    "Drei gleiche Lager": {"n": 3, "rho": 0.0, "spread": 0.0, "service": 0.75, "dist": DIST_NORMAL, "seed": DEFAULT_SEED},
    "Vier verschieden große Lager": {"n": 4, "rho": 0.0, "spread": 0.8, "service": 0.75, "dist": DIST_NORMAL, "seed": DEFAULT_SEED},
    "Hohe Korrelation (ρ = 0,8)": {"n": 4, "rho": 0.8, "spread": 0.4, "service": 0.75, "dist": DIST_NORMAL, "seed": DEFAULT_SEED},
    "Schwerer Tail (Szenarien)": {"n": 4, "rho": 0.3, "spread": 0.5, "service": 0.75, "dist": DIST_HEAVY, "seed": DEFAULT_SEED},
    "Großer Servicegrad": {"n": 4, "rho": 0.3, "spread": 0.5, "service": 0.99, "dist": DIST_NORMAL, "seed": DEFAULT_SEED},
}
DEFAULT_PRESET = "Drei gleiche Lager"
PRESET_HELP = {
    "Drei gleiche Lager": "Drei gleich große Lager mit unabhängiger Nachfrage (ρ = 0), Servicegrad 0,75, Normalverteilung. Der symmetrische Grundfall: kein Mitglied ist im Vorteil.",
    "Vier verschieden große Lager": "Vier Lager mit stark streuender Größe (Streuung 0,8), unabhängige Nachfrage, Normalverteilung. Zeigt, ob ungleiche Mitglieder gleich gut wegkommen.",
    "Hohe Korrelation (ρ = 0,8)": "Vier Lager, deren Nachfragen stark gleichlaufen (ρ = 0,8). Gleichlauf gleicht Schwankungen schlechter aus: der Gewinn des Bündelns sinkt.",
    "Schwerer Tail (Szenarien)": "Vier Lager mit schwerem Tail (Pareto-Index 2,5), gerechnet über 20.000 Zufallsszenarien statt über die Normalformel.",
    "Großer Servicegrad": "Servicegrad 0,99: Fehlmengen sind sehr teuer gegenüber Überbestand. Der Gewinn des Bündelns bleibt unter Normalverteilung derselbe, die Kosten steigen.",
}
