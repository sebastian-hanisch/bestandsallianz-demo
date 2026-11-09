"""Kern der Demo: Kostenspiel der Bestandsallianz, Aufteilungsregeln, Shapley-Wert und Kern (per LP, HiGHS).

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b). Ein Spiel ist die Kostenfunktion C über allen
Koalitionen als Feld der Länge 2^n (Index = Bitmaske, siehe bsa_costs.py). Aufteilungen sind Vektoren x der Länge n mit Summe C(N)."""

from dataclasses import dataclass
from math import factorial

import numpy as np

import bsa_constants as C
from bsa_costs import mask_matrix


@dataclass(eq=False)
class Game:
    n: int
    cost: np.ndarray            # Länge 2^n, cost[0] = 0

    @property
    def grand(self) -> float:
        return float(self.cost[2**self.n - 1])

    @property
    def single(self) -> np.ndarray:
        """Einzelkosten C({i}) = Stand-alone-Kosten je Mitglied."""
        return np.array([self.cost[1 << i] for i in range(self.n)])


def make_game(cost: np.ndarray) -> Game:
    """Aus einem Kostenfeld der Länge 2^n ein Spiel machen (Länge und leere Koalition prüfen)."""
    cost = np.asarray(cost, dtype=np.float64)
    n = int(round(np.log2(len(cost))))
    if 2**n != len(cost) or n < 1:
        raise ValueError("die Länge des Kostenfeldes muss 2^n mit n >= 1 sein")
    if cost[0] != 0.0:
        raise ValueError("die leere Koalition kostet nichts")
    return Game(n, cost)


def gain_of_pooling(game: Game) -> float:
    """Gewinn des Bündelns als Anteil der Summe der Einzelkosten: 1 - C(N) / Summe C({i}); 0, wenn nichts anfällt."""
    alone = float(game.single.sum())
    return 0.0 if alone <= 0 else 1.0 - game.grand / alone


def shapley_value(game: Game) -> np.ndarray:
    """Shapley-Wert über die Teilmengen-Formel: phi_i = Summe über S ohne i von s! (n-s-1)! / n! * (C(S + i) - C(S)), vektorisiert über alle Masken."""
    n = game.n
    masks = mask_matrix(n)
    size = masks.sum(axis=1).astype(int)
    weight = np.array([factorial(s) * factorial(n - s - 1) / factorial(n) if s < n else 0.0 for s in range(n + 1)])
    s_all = np.arange(2**n)
    phi = np.zeros(n)
    for i in range(n):
        without = s_all[(s_all >> i) & 1 == 0]
        phi[i] = float(np.sum(weight[size[without]] * (game.cost[without | (1 << i)] - game.cost[without])))
    return phi


def allocation_equal(game: Game) -> np.ndarray:
    """Gleichteilung: jedes Mitglied zahlt C(N) / n."""
    return np.full(game.n, game.grand / game.n)


def allocation_proportional(game: Game, weights) -> np.ndarray:
    """Proportional zu positiven Gewichten; bei Gewichtssumme 0 Gleichteilung."""
    w = np.asarray(weights, dtype=np.float64)
    total = float(w.sum())
    return allocation_equal(game) if total <= 0 else game.grand * w / total


def allocate(rule: str, game: Game, mu) -> np.ndarray:
    """Aufteilung nach Regelname; `mu` sind die mittleren Nachfragen (Gewichte der Regel „Proportional zum Mittelwert“)."""
    if rule == C.RULE_BASELINE:
        return game.single.copy()
    if rule == C.RULE_EQUAL:
        return allocation_equal(game)
    if rule == C.RULE_MEAN:
        return allocation_proportional(game, mu)
    if rule == C.RULE_SINGLE:
        return allocation_proportional(game, game.single)
    if rule == C.RULE_SHAPLEY:
        return shapley_value(game)
    raise ValueError(f"unbekannte Regel {rule!r}")


def excess(game: Game, x) -> np.ndarray:
    """Überschuss x(S) - C(S) jeder Koalition (Länge 2^n); positiv heißt: S zahlt in der Allianz mehr, als sie allein zahlen würde."""
    return mask_matrix(game.n) @ np.asarray(x, dtype=np.float64) - game.cost


def max_excess(game: Game, x) -> tuple[float, int]:
    """Größte Unzufriedenheit unter den echten Koalitionen (nicht leer, nicht N): (Überschuss, Maske). Bei n = 1 gibt es keine echte Koalition."""
    if game.n < 2:
        return 0.0, 0
    e = excess(game, x)[1 : 2**game.n - 1]
    k = int(np.argmax(e))                       # bei Gleichstand die kleinste Maske
    return float(e[k]), k + 1


def is_in_core(game: Game, x, tol: float = C.TOL) -> bool:
    """x ist eine Aufteilung (Summe C(N)) und keine echte Koalition würde austreten wollen: Überschuss höchstens tol * C(N)."""
    x = np.asarray(x, dtype=np.float64)
    if abs(float(x.sum()) - game.grand) > 1e-7 * max(game.grand, 1.0):
        return False
    return max_excess(game, x)[0] <= tol * game.grand


def check_feasible(game: Game, x) -> bool:
    """Unabhängiger Prüfer der Aufteilung: endliche Werte und Summe gleich C(N)."""
    x = np.asarray(x, dtype=np.float64)
    return len(x) == game.n and bool(np.all(np.isfinite(x))) and abs(float(x.sum()) - game.grand) <= 1e-7 * max(game.grand, 1.0)


def member_savings(game: Game, x) -> np.ndarray:
    """Ersparnis je Mitglied als Anteil der Einzelkosten: (C({i}) - x_i) / C({i})."""
    single = game.single
    return np.where(single > 0, (single - np.asarray(x)) / np.where(single > 0, single, 1.0), 0.0)


def _proper_rows(game: Game):
    """Zeilen (Maske, Koalitionskosten) der echten Koalitionen für die LPs."""
    masks = mask_matrix(game.n)[1 : 2**game.n - 1]
    return masks, game.cost[1 : 2**game.n - 1]


def least_core_epsilon(game: Game) -> tuple[float, np.ndarray]:
    """Kleinstes epsilon, für das x(S) <= C(S) + epsilon für alle echten Koalitionen S und Summe x = C(N) lösbar ist; dazu eine Aufteilung.

    Der Kern ist genau dann nicht leer, wenn epsilon* <= 0. Gerechnet wird mit auf C(N) normierten Kosten (bessere Konditionierung)."""
    from scipy.optimize import linprog

    n, grand = game.n, game.grand
    if n < 2 or grand <= 0:
        return 0.0, np.full(n, grand / n if n else 0.0)
    masks, rhs = _proper_rows(game)
    a_ub = np.hstack([masks, -np.ones((len(masks), 1))])
    res = linprog(
        np.append(np.zeros(n), 1.0), A_ub=a_ub, b_ub=rhs / grand, A_eq=[np.append(np.ones(n), 0.0)], b_eq=[1.0],
        bounds=[(None, None)] * (n + 1), method="highs",
    )
    if res.status != 0:
        raise RuntimeError(f"LP des Kerns nicht gelöst: {res.message}")
    return float(res.x[n]) * grand, res.x[:n] * grand


def core_is_nonempty(game: Game, tol: float = C.TOL) -> bool:
    """Kern nicht leer genau dann, wenn epsilon* <= 0 (bis auf die Toleranz tol * C(N))."""
    return least_core_epsilon(game)[0] <= tol * game.grand


def core_range(game: Game) -> tuple[np.ndarray, np.ndarray] | None:
    """Kern-Spielraum: kleinster und größter Beitrag x_i je Mitglied über alle Aufteilungen im Kern (2n LPs); None, wenn der Kern leer ist."""
    from scipy.optimize import linprog

    n, grand = game.n, game.grand
    if not core_is_nonempty(game):
        return None
    if n < 2:
        return np.array([grand]), np.array([grand])
    masks, rhs = _proper_rows(game)
    rhs = rhs / grand + C.TOL                                   # die Toleranz hält den Grenzfall epsilon* = 0 lösbar
    lo, hi = np.zeros(n), np.zeros(n)
    for i in range(n):
        for sign, out in ((1.0, lo), (-1.0, hi)):
            c = np.zeros(n)
            c[i] = sign
            res = linprog(c, A_ub=masks, b_ub=rhs, A_eq=[np.ones(n)], b_eq=[1.0], bounds=[(None, None)] * n, method="highs")
            if res.status != 0:
                raise RuntimeError(f"LP des Kern-Spielraums nicht gelöst: {res.message}")
            out[i] = res.x[i] * grand
    return lo, hi


def is_cost_concave(game: Game, tol: float = C.TOL) -> bool:
    """Konkav (Grenzkosten fallen mit der Allianzgröße): C(S+i+j) - C(S+i) - C(S+j) + C(S) <= 0 für alle S und alle Paare i != j außerhalb von S."""
    s_all = np.arange(2**game.n)
    c = game.cost
    for i in range(game.n):
        for j in range(i + 1, game.n):
            s = s_all[((s_all >> i) & 1 == 0) & ((s_all >> j) & 1 == 0)]
            if np.any(c[s | (1 << i) | (1 << j)] - c[s | (1 << i)] - c[s | (1 << j)] + c[s] > tol * game.grand):
                return False
    return True


def shapley_trace(game: Game, member: int) -> list[tuple[tuple[int, ...], float]]:
    """Beispielspur für die App: je Reihenfolge, in der die Lager der Allianz beitreten, die Zusatzkosten von `member` (nur für kleine n gedacht, n! Zeilen)."""
    from itertools import permutations

    rows = []
    for order in permutations(range(game.n)):
        before = 0
        for i in order:
            if i == member:
                break
            before |= 1 << i
        rows.append((order, float(game.cost[before | (1 << member)] - game.cost[before])))
    return rows
