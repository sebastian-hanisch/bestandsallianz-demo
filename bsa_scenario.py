"""Szenario: n Lager mit gemeinsam verteilter Nachfrage (Normal, Lognormal, Pareto) und gleichen Stückkosten.

Alle Zufallszahlen kommen aus SplitMix64 (`bsa_rng.py`): gleiche Eingabe, gleiche Instanz, auf jeder Plattform.
Die Abhängigkeit ist eine Gleichkorrelation rho der zugrunde liegenden Standardnormalvariablen (Gauß-Copula); bei der Normalverteilung ist rho
die Korrelation der Nachfragen selbst."""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri

import bsa_constants as C
from bsa_rng import uniform_block


@dataclass(frozen=True)
class Instance:
    mu: tuple[float, ...]       # mittlere Nachfrage je Lager
    sd: tuple[float, ...]       # Standardabweichung je Lager (Normalfall)
    shape: tuple[float, ...]    # Lognormal-Formparameter je Lager
    rho: float
    service: float              # Servicegrad c_u / (c_u + c_o)
    dist: str
    seed: int

    @property
    def n(self) -> int:
        return len(self.mu)

    @property
    def cu_plus_co(self) -> float:
        """c_u + c_o = c_o / (1 - Servicegrad), in Einheiten von c_o."""
        return C.CO / (1.0 - self.service)


def make_instance(n: int, rho: float, spread: float, service: float, dist: str, seed: int) -> Instance:
    """Deterministisch aus den Parametern. Größe_i = 1 + spread * (2 u_i - 1); mu_i = MU_BASE * Größe_i;
    sd_i = CV_NORMAL * mu_i * (1 + 0,5 * spread * (2 v_i - 1)); u, v gleichverteilt (SplitMix64)."""
    if dist not in C.DISTS:
        raise ValueError(f"unbekannte Verteilung {dist!r}")
    if not 0.0 < service < 1.0:
        raise ValueError("Servicegrad muss zwischen 0 und 1 liegen")
    if not 0.0 <= rho < 1.0 or not 0.0 <= spread < 1.0:
        raise ValueError("rho und spread müssen in [0, 1) liegen")
    u = uniform_block(seed, 2 * n)
    size = 1.0 + spread * (2.0 * u[:n] - 1.0)
    factor = 1.0 + 0.5 * spread * (2.0 * u[n:] - 1.0)
    mu = C.MU_BASE * size
    sd = C.CV_NORMAL * mu * factor
    shape = C.LN_SHAPE * factor
    return Instance(tuple(float(x) for x in mu), tuple(float(x) for x in sd), tuple(float(x) for x in shape), float(rho), float(service), dist, int(seed))


def covariance(inst: Instance) -> np.ndarray:
    """Kovarianzmatrix der Normal-Nachfrage: sd_i sd_j rho für i != j, sd_i^2 auf der Diagonale."""
    sd = np.array(inst.sd)
    cov = inst.rho * np.outer(sd, sd)
    np.fill_diagonal(cov, sd**2)
    return cov


def standard_normals(n: int, rho: float, seed: int, m: int = C.SCENARIOS) -> np.ndarray:
    """m x n gleichkorrelierte Standardnormalvariablen: Z_i = sqrt(rho) F + sqrt(1 - rho) E_i mit gemeinsamem Faktor F."""
    u = uniform_block(seed + C.SCEN_SEED_OFFSET, m * (n + 1)).reshape(m, n + 1)
    u = np.clip(u, 2.0**-54, 1.0 - 2.0**-53)
    g = ndtri(u)
    return np.sqrt(rho) * g[:, :1] + np.sqrt(1.0 - rho) * g[:, 1:]


def scenario_demands(inst: Instance, m: int = C.SCENARIOS) -> np.ndarray:
    """m x n Nachfrage-Szenarien der gewählten Verteilung (alle Koalitionen nutzen dieselben Szenarien)."""
    z = standard_normals(inst.n, inst.rho, inst.seed, m)
    mu = np.array(inst.mu)
    if inst.dist == C.DIST_NORMAL:
        return mu + np.array(inst.sd) * z
    if inst.dist == C.DIST_LOGNORMAL:
        lam = np.array(inst.shape)
        return mu * np.exp(lam * z - lam**2 / 2.0)
    a = C.PARETO_ALPHA                                  # schwerer Tail: Pareto mit Mittel mu, Quantilfunktion xm (1 - U)^(-1/a)
    return mu * (a - 1.0) / a * ndtr(-z) ** (-1.0 / a)
