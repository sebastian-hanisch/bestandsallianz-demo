"""Koalitionskosten C(S): Erwartungskosten des optimalen Zeitungsjungen-Bestands für die Summe der Nachfragen der Mitglieder von S.

Eine Koalition wird als Bitmaske gespeichert (Bit i gesetzt: Lager i gehört dazu); `C[0] = 0` für die leere Koalition.
Zwei Wege, dieselbe Größe zu berechnen: die Normalformel `(c_u + c_o) phi(z) sigma_S` und der empirische Erwartungswert über Szenarien."""

import numpy as np
from scipy.special import ndtri

import bsa_constants as C
from bsa_scenario import Instance, covariance, scenario_demands


def mask_matrix(n: int) -> np.ndarray:
    """2^n x n Matrix: Zeile s hat in Spalte i eine 1, wenn Bit i von s gesetzt ist."""
    s = np.arange(2**n)
    return ((s[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)


def normal_newsvendor_cost(sigma_sum: np.ndarray | float, service: float) -> np.ndarray | float:
    """Erwartungskosten des optimalen Bestands bei Normalnachfrage mit Standardabweichung sigma_sum (Summe über die Koalition).

    Mit z = Phi^-1(Servicegrad): (c_u + c_o) phi(z) sigma; die Einheit ist c_o, c_u + c_o = c_o / (1 - Servicegrad)."""
    z = ndtri(service)
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return C.CO / (1.0 - service) * phi * np.asarray(sigma_sum)


def coalition_costs_cov(cov: np.ndarray, service: float) -> np.ndarray:
    """Alle 2^n Koalitionskosten bei gemeinsamer Normalnachfrage mit beliebiger Kovarianzmatrix: sigma_S^2 = 1_S' Sigma 1_S, vektorisiert über alle Masken."""
    masks = mask_matrix(cov.shape[0])
    var = np.einsum("si,ij,sj->s", masks, cov, masks)
    return normal_newsvendor_cost(np.sqrt(np.maximum(var, 0.0)), service)


def coalition_costs_normal(inst: Instance) -> np.ndarray:
    """Koalitionskosten der Instanz bei Normalnachfrage (Gleichkorrelation rho)."""
    return coalition_costs_cov(covariance(inst), inst.service)


def empirical_newsvendor_cost(totals: np.ndarray, service: float) -> np.ndarray:
    """Empirisches Optimum je Spalte: Bestand q = kleinster Wert mit Anteil >= Servicegrad, Kosten = Mittel von c_u (T - q)+ + c_o (q - T)+.

    `totals` hat die Form (m, k): m Szenarien, k Koalitionen. Die Rechnung läuft über die sortierten Spalten, ohne Schleife über Szenarien."""
    m = totals.shape[0]
    cu = C.CO * service / (1.0 - service)
    t = np.sort(totals, axis=0)
    j = min(max(int(np.ceil(m * service)) - 1, 0), m - 1)
    q = t[j]
    below = (j + 1) * q - t[: j + 1].sum(axis=0)                 # Summe von (q - T) über die Szenarien bis zum Index j
    above = t[j + 1 :].sum(axis=0) - (m - j - 1) * q             # Summe von (T - q) über die übrigen
    return (cu * above + C.CO * below) / m


def coalition_costs_scenarios(demands: np.ndarray, service: float, block: int = 32) -> np.ndarray:
    """Alle 2^n Koalitionskosten als empirischer Erwartungswert über dieselben Szenarien (in Spaltenblöcken, um den Speicher klein zu halten)."""
    n = demands.shape[1]
    masks = mask_matrix(n)
    out = np.zeros(2**n)
    for start in range(1, 2**n, block):
        stop = min(start + block, 2**n)
        out[start:stop] = empirical_newsvendor_cost(demands @ masks[start:stop].T, service)
    return out


def coalition_costs(inst: Instance) -> np.ndarray:
    """Koalitionskosten der Instanz: Formel bei Normalverteilung, Szenarien sonst."""
    if inst.dist == C.DIST_NORMAL:
        return coalition_costs_normal(inst)
    return coalition_costs_scenarios(scenario_demands(inst), inst.service)
