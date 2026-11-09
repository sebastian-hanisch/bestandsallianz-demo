"""Gemeinsame Fixtures: die Mini-Spiele, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bsa_algorithm as A  # noqa: E402


def newsvendor_k(service: float) -> float:
    """Faktor k mit C(S) = k * sigma_S bei Normalnachfrage: (c_u + c_o) * phi(z) mit c_o = 1 und c_u + c_o = 1 / (1 - Servicegrad), über scipy.stats gerechnet."""
    return float(norm.pdf(norm.ppf(service)) / (1.0 - service))


def make_cost_game(costs: dict[tuple[int, ...], float], n: int) -> A.Game:
    """Spiel aus einem Wörterbuch {Mitglieder-Tupel: Kosten}."""
    cost = np.zeros(2**n)
    for members, value in costs.items():
        cost[sum(1 << i for i in members)] = value
    return A.make_game(cost)


@pytest.fixture
def empty_core_game():
    """Drei Lager, jede Einzelkoalition und jedes Paar kostet 10, alle drei zusammen 16. Von Hand: die drei Paar-Ungleichungen ergeben
    2 * x(N) <= 30 + 3 * epsilon, also epsilon* = (32 - 30) / 3 = 2/3 > 0: der Kern ist leer."""
    pairs = {(0, 1): 10.0, (0, 2): 10.0, (1, 2): 10.0}
    singles = {(0,): 10.0, (1,): 10.0, (2,): 10.0}
    return make_cost_game({**singles, **pairs, (0, 1, 2): 16.0}, 3)


@pytest.fixture
def mini():
    """Zwei Lager, Normalnachfrage mit sigma = 3 und 4, unabhängig, Servicegrad 0,75. Von Hand mit k aus `newsvendor_k`:
    C({1}) = 3k, C({2}) = 4k, C(N) = 5k (sigma_N = sqrt(9 + 16)). Gewinn 1 - 5/7, Shapley (2k, 3k), epsilon* = -k, Kern x_1 in [k, 3k]."""
    from bsa_costs import coalition_costs_cov

    return A.make_game(coalition_costs_cov(np.diag([9.0, 16.0]), 0.75)), newsvendor_k(0.75)
