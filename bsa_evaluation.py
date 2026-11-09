"""Kennzahlen, Sweeps und Verdikte der Bestandsallianz. Verdikte sind reine Funktionen `verdict_*(…) -> (Art, Text)`."""

from dataclasses import dataclass

import numpy as np

import bsa_algorithm as A
import bsa_constants as C
from bsa_costs import coalition_costs
from bsa_rng import uniform_block
from bsa_scenario import Instance, make_instance


@dataclass(frozen=True, kw_only=True)
class Settings:
    n: int = C.DEFAULT_N
    rho: float = C.DEFAULT_RHO
    spread: float = C.DEFAULT_SPREAD
    service: float = C.DEFAULT_SERVICE
    dist: str = C.DEFAULT_DIST
    seed: int = C.DEFAULT_SEED


@dataclass(eq=False)
class Analysis:
    inst: Instance
    game: A.Game
    gain: float                      # Anteil, 1 - C(N) / Summe der Einzelkosten
    eps: float                       # kleinstes epsilon des Kern-LP; Kern nicht leer genau dann, wenn eps <= 0
    core_nonempty: bool
    rules: dict                      # Regelname -> Aufteilung (Vektor)
    in_core: dict                    # Regelname -> bool (nur Aufteilungsregeln)
    excess_rel: dict                 # Regelname -> größte Unzufriedenheit einer echten Koalition in Anteilen von deren Kosten
    savings: dict                    # Regelname -> Ersparnis je Mitglied als Anteil der Einzelkosten
    core_lo: np.ndarray | None       # Kern-Spielraum je Mitglied (None bei leerem Kern)
    core_hi: np.ndarray | None


def build_game(inst: Instance) -> A.Game:
    return A.make_game(coalition_costs(inst))


def inst_of(s: Settings) -> Instance:
    return make_instance(s.n, s.rho, s.spread, s.service, s.dist, s.seed)


def worst_coalition_excess_rel(game: A.Game, x) -> float:
    """Größte Unzufriedenheit unter den echten Koalitionen, bezogen auf deren Einzelkosten C(S) (Anteil)."""
    value, mask = A.max_excess(game, x)
    return 0.0 if mask == 0 else value / float(game.cost[mask])


def analyse(s: Settings, with_range: bool = True) -> Analysis:
    inst = inst_of(s)
    game = build_game(inst)
    eps, _ = A.least_core_epsilon(game)
    nonempty = eps <= C.TOL * game.grand
    rules = {rule: A.allocate(rule, game, inst.mu) for rule in C.RULES}
    for rule in C.ALLOCATION_RULES:
        if not A.check_feasible(game, rules[rule]):
            raise RuntimeError(f"Aufteilung {rule!r} ist nicht zulässig (Summe ungleich C(N))")
    rng = A.core_range(game) if (with_range and nonempty) else None
    return Analysis(
        inst, game, A.gain_of_pooling(game), eps, nonempty, rules,
        {r: A.is_in_core(game, rules[r]) for r in C.ALLOCATION_RULES},
        {r: worst_coalition_excess_rel(game, rules[r]) for r in C.ALLOCATION_RULES},
        {r: A.member_savings(game, rules[r]) for r in C.RULES},
        None if rng is None else rng[0], None if rng is None else rng[1],
    )


def rule_rows(a: Analysis) -> list[dict]:
    """Zeilen der Regeltabelle: Ersparnis je Mitglied (kleinste und größte, in Prozent der Einzelkosten), Kern-Zugehörigkeit, größte Unzufriedenheit."""
    rows = []
    for rule in C.RULES:
        s = 100 * a.savings[rule]
        baseline = rule == C.RULE_BASELINE
        rows.append({
            "Regel": rule, "Kleinste Ersparnis (%)": float(s.min()), "Größte Ersparnis (%)": float(s.max()),
            "Im Kern": "keine Aufteilung" if baseline else ("ja" if a.in_core[rule] else "nein"),
            "Größte Unzufriedenheit einer Koalition (% ihrer Kosten)": None if baseline else 100 * a.excess_rel[rule],
        })
    return rows


def core_rows(a: Analysis) -> list[dict]:
    """Zeilen der Kern-Spielraum-Tabelle: Einzelkosten, kleinster und größter Beitrag im Kern, Shapley-Beitrag je Lager (None bei leerem Kern)."""
    if a.core_lo is None:
        return []
    single = a.game.single
    return [
        {"Lager": f"Lager {i + 1}", "Einzelkosten": float(single[i]), "Kern: kleinster Beitrag": float(a.core_lo[i]), "Kern: größter Beitrag": float(a.core_hi[i]),
         "Shapley-Wert": float(a.rules[C.RULE_SHAPLEY][i])}
        for i in range(a.game.n)
    ]


# --- Experiment 1: Gewinn des Bündelns über Lagerzahl und Korrelation (Normalformel, 20 feste Instanzen je Zelle) ----------------------------------

def gain_normal(inst: Instance) -> float:
    """Gewinn des Bündelns bei Normalnachfrage in geschlossener Form: 1 - sigma_N / Summe sigma_i (der Servicegrad kürzt sich heraus)."""
    sd = np.array(inst.sd)
    var_n = (1.0 - inst.rho) * float((sd**2).sum()) + inst.rho * float(sd.sum()) ** 2
    return 1.0 - float(np.sqrt(var_n)) / float(sd.sum())


def gain_table(spread: float, service: float, ns=C.N_SWEEP, rhos=C.RHO_SWEEP, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Gewinn des Bündelns je Zelle (n, rho): Mittel, Minimum und Maximum über die festen Instanzen `seeds` (andere Größen je Seed, Streuung wie im Regler)."""
    rows = []
    for n in ns:
        for rho in rhos:
            values = [gain_normal(make_instance(n, rho, spread, service, C.DIST_NORMAL, seed)) for seed in seeds]
            rows.append({"n": n, "rho": rho, "mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))})
    return rows


# --- Experimente 2 bis 4: Zufallsinstanzen ------------------------------------------------------------------------------------------------------

def random_settings(index: int, dist: str, base: int = C.RANDOM_SEED_BASE) -> Settings:
    """Die Instanz Nr. `index` der Zufallsreihe: Lagerzahl, rho, Streuung und Servicegrad aus SplitMix64 gezogen, Instanz-Seed = base + index."""
    u = uniform_block(base + 7919 * index, 4)
    n_max = C.RANDOM_N_MAX if dist == C.DIST_NORMAL else C.RANDOM_N_MAX_SCENARIO
    n = C.RANDOM_N_MIN + min(int(u[0] * (n_max - C.RANDOM_N_MIN + 1)), n_max - C.RANDOM_N_MIN)
    return Settings(n=n, rho=round(float(u[1]) * 0.9, 2), spread=round(float(u[2]) * 0.9, 2), service=round(0.55 + float(u[3]) * 0.4, 2), dist=dist, seed=base + index)


def random_records(dist: str, count: int, base: int = C.RANDOM_SEED_BASE) -> list[dict]:
    """Messreihe über `count` Zufallsinstanzen: je Instanz Gewinn, epsilon, Kern-Zugehörigkeit je Regel, Kern-Spielraum und Austrittsanreize."""
    records = []
    for index in range(count):
        s = random_settings(index, dist, base)
        a = analyse(s)
        single = a.game.single
        rec = {
            "n": s.n, "rho": s.rho, "spread": s.spread, "service": s.service, "gain": a.gain, "eps": a.eps, "eps_rel": a.eps / a.game.grand,
            "core_nonempty": a.core_nonempty, "in_core": dict(a.in_core), "excess_rel": dict(a.excess_rel),
            "width_rel": None if a.core_lo is None else float(np.mean((a.core_hi - a.core_lo) / single)),
            "mu": a.inst.mu, "single_exit": {r: (a.rules[r] - single) / single for r in C.ALLOCATION_RULES},
        }
        records.append(rec)
    return records


def count_in_core(records: list[dict]) -> dict:
    """Je Aufteilungsregel: in wie vielen Instanzen liegt die Aufteilung im Kern?"""
    return {rule: sum(1 for r in records if r["in_core"][rule]) for rule in C.ALLOCATION_RULES}


def count_empty_core(records: list[dict]) -> int:
    """Zahl der Instanzen mit leerem Kern (epsilon* > Toleranz)."""
    return sum(1 for r in records if not r["core_nonempty"])


def worst_violation(records: list[dict], rule: str) -> float:
    """Größte Unzufriedenheit (Anteil der Koalitionskosten) einer echten Koalition über alle Instanzen für die Regel; 0, wenn die Regel stets im Kern liegt."""
    return max((r["excess_rel"][rule] for r in records if not r["in_core"][rule]), default=0.0)


RHO_BINS = ((0.0, 0.3), (0.3, 0.6), (0.6, 0.9))     # Klassen von rho für Experiment 4 (obere Grenze der letzten Klasse inklusive)


def exit_summary(records: list[dict], rule: str) -> dict:
    """Einzelaustritt unter einer Regel: in wie vielen Instanzen zahlt mindestens ein Mitglied mehr als allein (x_i > C({i}))?

    Je Instanz zählt das Mitglied mit dem größten Anreiz (x_i - C_i) / C_i; es heißt „kleinstes Lager“, wenn seine mittlere Nachfrage das Minimum ist
    (bei Gleichstand zählen alle), „größtes“ entsprechend. Dazu je rho-Klasse die Zahl der Instanzen und der Austritte."""
    total, exits, smallest, largest = len(records), 0, 0, 0
    by_rho = []
    for lo, hi in RHO_BINS:
        in_bin = [r for r in records if lo <= r["rho"] < hi or (hi == RHO_BINS[-1][1] and r["rho"] == hi)]
        leaving = [r for r in in_bin if (r["single_exit"][rule] > C.TOL).any()]
        by_rho.append({"lo": lo, "hi": hi, "instances": len(in_bin), "exits": len(leaving)})
    for r in records:
        incentive = np.asarray(r["single_exit"][rule])
        if not (incentive > C.TOL).any():
            continue
        exits += 1
        k = int(np.argmax(incentive))
        mu = np.asarray(r["mu"])
        smallest += int(mu[k] <= mu.min())
        largest += int(mu[k] >= mu.max())
    return {"instances": total, "exits": exits, "smallest": smallest, "largest": largest, "by_rho": by_rho}


# --- Verdikte (reine Funktionen: (Art, Text)) -----------------------------------------------------------------------------------------------

def de(x: float, digits: int = 1) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


MIN_GAIN = 0.005    # unter 0,5 % der Einzelkosten gilt der Gewinn des Bündelns als kaum vorhanden


def de_small(x: float) -> str:
    """Prozentwert mit zwei Stellen, bei kleinen Werten (unter 0,01) mit vier, damit eine winzige Verletzung nicht als 0,00 erscheint."""
    return de(x, 2 if abs(x) >= 0.01 else 4)


def verdict_alliance(*, gain: float, eps_rel: float, core_nonempty: bool, shapley_in_core: bool, min_saving: float, min_member: int,
                     shapley_excess_rel: float = 0.0) -> tuple[str, str]:
    """Kernfrage: lohnt sich die Allianz für jeden, und gibt es eine Aufteilung ohne Austrittsgrund?

    `gain`: Anteil der Einzelkosten, `eps_rel`: epsilon* in Anteilen der Allianzkosten, `min_saving`: kleinste Ersparnis eines Mitglieds
    unter dem Shapley-Wert (Anteil der Einzelkosten), `min_member`: dessen Index (ab 0), `shapley_excess_rel`: größte Überschreitung einer Koalition unter dem
    Shapley-Wert in Anteilen der Allianzkosten (nur bei Verletzung gelesen)."""
    if not core_nonempty:
        return "warning", (f"**Der Kern ist leer:** ε* = {de(100 * eps_rel, 2)} % der Allianzkosten ist größer als 0. Keine Aufteilung schützt vor dem Austritt "
                           "mindestens einer Teilgruppe. Das Bündeln spart "
                           f"{de(100 * gain)} % der Summe der Einzelkosten.")
    if gain < MIN_GAIN:
        return "info", (f"Das Bündeln spart hier kaum etwas: {de(100 * gain, 2)} % der Summe der Einzelkosten. Der Kern ist nicht leer, "
                        "aber es gibt fast nichts zu verteilen.")
    if not shapley_in_core:
        return "warning", (f"Das Bündeln spart {de(100 * gain)} % der Summe der Einzelkosten und der Kern ist nicht leer (ε* = {de(100 * eps_rel, 2)} %), "
                           f"aber der Shapley-Wert liegt nicht im Kern: die am stärksten benachteiligte Teilgruppe würde allein {de_small(100 * shapley_excess_rel)} % der Allianzkosten sparen.")
    if min_saving <= 0:
        return "info", (f"Das Bündeln spart {de(100 * gain)} % der Summe der Einzelkosten und der Shapley-Wert liegt im Kern, aber Lager {min_member + 1} spart unter dem "
                        "Shapley-Wert nichts gegenüber allein.")
    return "success", (f"**Ja, im Modell lohnt sich die Allianz für alle.** Sie spart {de(100 * gain)} % der Summe der Einzelkosten. Der Kern ist nicht leer "
                       f"(ε* = {de(100 * eps_rel, 2)} % der Allianzkosten, also kleiner als 0), und der Shapley-Wert liegt im Kern: keine Teilgruppe kommt allein günstiger weg. "
                       f"Die kleinste Ersparnis hat Lager {min_member + 1} mit {de(100 * min_saving)} % seiner Einzelkosten.")


def verdict_gain(rows: list[dict]) -> tuple[str, str]:
    """Experiment 1: kleinster und größter Mittelwert der Tabelle, kleinster Einzelwert, Richtung in n und in rho als Zählung über alle Reihen."""
    ns = sorted({r["n"] for r in rows})
    rhos = sorted({r["rho"] for r in rows})
    cell = {(r["n"], r["rho"]): r for r in rows}
    low = min(rows, key=lambda r: r["mean"])
    high = max(rows, key=lambda r: r["mean"])
    smallest = min(r["min"] for r in rows)
    falling = sum(all(cell[(n, a)]["mean"] > cell[(n, b)]["mean"] for a, b in zip(rhos, rhos[1:])) for n in ns)
    rising = sum(all(cell[(a, rho)]["mean"] < cell[(b, rho)]["mean"] for a, b in zip(ns, ns[1:])) for rho in rhos)
    return "info", (
        f"Der mittlere Gewinn liegt zwischen {de(100 * low['mean'])} % (n = {low['n']}, ρ = {de(low['rho'], 2)}) und {de(100 * high['mean'])} % "
        f"(n = {high['n']}, ρ = {de(high['rho'], 2)}); der kleinste Einzelwert über alle {len(rows)} Zellen ist {de(100 * smallest)} %. "
        f"Mit wachsendem ρ sinkt der Mittelwert bei {falling} von {len(ns)} Lagerzahlen, mit wachsender Lagerzahl steigt er bei {rising} von {len(rhos)} Werten von ρ."
    )


def verdict_rules(counts: dict, instances: int, dist: str) -> tuple[str, str]:
    """Experiment 2: je Regel die Zahl der Instanzen, in denen die Aufteilung im Kern liegt."""
    parts = ", ".join(f"{rule} {counts[rule]}" for rule in (C.RULE_SHAPLEY, C.RULE_SINGLE, C.RULE_MEAN, C.RULE_EQUAL))
    head = f"Im Kern lagen bei {instances} Zufallsinstanzen ({dist}): {parts}."
    if instances > 0 and counts[C.RULE_SHAPLEY] == instances:
        return "success", head + " Der Shapley-Wert lag in jeder gezogenen Instanz im Kern; das ist eine Zählung über diese Reihe, kein Beweis. An den Reglerrändern der Seitenleiste kann er knapp danebenliegen (siehe Annahmen, Grenzen und bekannte Probleme)."
    return "warning", head + " Der Shapley-Wert lag nicht in jeder gezogenen Instanz im Kern."


def verdict_core_empty(empty: int, instances: int, dist: str, worst_eps_rel: float) -> tuple[str, str]:
    """Experiment 3: Zahl der Instanzen mit leerem Kern und größtes epsilon* in Anteilen der Allianzkosten (negativ heißt: Kern nicht leer)."""
    base = (f"Bei gleichen Stückkosten war der Kern in {empty} von {instances} Zufallsinstanzen leer ({dist}); "
            f"das größte ε* war {de(100 * worst_eps_rel, 2)} % der Allianzkosten.")
    if empty == 0:
        return "success", base + " Das ist eine Zählung, kein Beweis."
    return "warning", base + " Der Kern war also nicht immer nichtleer: Rundung und Toleranz prüfen."


def verdict_exit(summary: dict, rule: str) -> tuple[str, str]:
    """Experiment 4: Einzelaustritt unter einer Regel, wer geht, und die Zählung je rho-Klasse."""
    n_exit, total = summary["exits"], summary["instances"]
    if n_exit == 0:
        return "success", f"Unter der Regel „{rule}“ zahlte in keiner der {total} Instanzen ein Mitglied mehr als allein: es gibt hier keinen Einzelaustritt."
    bins = "; ".join(f"ρ von {de(b['lo'], 1)} bis {de(b['hi'], 1)}: {b['exits']} von {b['instances']}" for b in summary["by_rho"])
    return "warning", (f"Unter der Regel „{rule}“ zahlte in {n_exit} von {total} Instanzen mindestens ein Mitglied mehr als allein. "
                       f"Das Mitglied mit dem größten Anreiz hatte in {summary['smallest']} dieser Fälle die kleinste mittlere Nachfrage, in {summary['largest']} die größte. "
                       f"Einzelaustritte je ρ-Klasse: {bins}.")
