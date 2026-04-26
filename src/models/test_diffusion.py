"""
Runnable sanity checks for ``simulate_ic`` (not a full pytest suite).

Run: ``python -m src.models.test_diffusion`` from project root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx

# Project root on path when executed as module
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.diffusion import simulate_ic


def _chain_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2)])
    return g


def test_chain_p1_reaches_all() -> None:
    g = _chain_graph()
    out = simulate_ic(g, [0], blocked_nodes=[], p=1.0, n_runs=1, random_seed=0)
    assert out["infected_counts"][0] == 3, out


def test_chain_blocked_middle_stops() -> None:
    g = _chain_graph()
    out = simulate_ic(g, [0], blocked_nodes={1}, p=1.0, n_runs=1, random_seed=1)
    assert out["infected_counts"][0] == 1, out


def test_empty_seeds_zero_infected() -> None:
    g = _chain_graph()
    out = simulate_ic(g, [], blocked_nodes=[], p=1.0, n_runs=3, random_seed=2)
    assert out["infected_counts"] == [0, 0, 0]
    assert out["mean_infected"] == 0.0


def test_p0_only_seeds() -> None:
    g = _chain_graph()
    out = simulate_ic(g, [0, 1], blocked_nodes=[], p=0.0, n_runs=5, random_seed=3)
    assert all(c == 2 for c in out["infected_counts"]), out["infected_counts"]


def test_blocked_seed_removed() -> None:
    g = _chain_graph()
    out = simulate_ic(g, [0], blocked_nodes={0}, p=1.0, n_runs=1, random_seed=4)
    assert out["infected_counts"][0] == 0


def test_reproducibility() -> None:
    g = nx.erdos_renyi_graph(30, 0.15, seed=999)
    g = nx.Graph(g)
    seeds = [0, 1, 2]
    blocked = {5, 6}
    a = simulate_ic(
        g,
        seeds,
        blocked_nodes=blocked,
        p=0.12,
        n_runs=40,
        random_seed=12345,
    )
    b = simulate_ic(
        g,
        seeds,
        blocked_nodes=blocked,
        p=0.12,
        n_runs=40,
        random_seed=12345,
    )
    assert a["infected_counts"] == b["infected_counts"]
    assert a["mean_infected"] == b["mean_infected"]


def test_communities_key_present() -> None:
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 3)])
    comm = {0: 0, 1: 0, 2: 1, 3: 1}
    out = simulate_ic(
        g,
        [0],
        blocked_nodes=[],
        p=1.0,
        n_runs=2,
        random_seed=7,
        communities=comm,
    )
    assert out["community_mean_infection_rate"] is not None
    assert "0" in out["community_mean_infection_rate"]
    assert out["mean_worst_community_infection_rate"] is not None
    assert out["mean_worst_community_infection_rate"] >= 0.0


def test_harm_metrics_present() -> None:
    g = _chain_graph()
    out = simulate_ic(
        g,
        [0],
        blocked_nodes=[],
        p=1.0,
        n_runs=1,
        random_seed=8,
        scenario_harm_weight=2.5,
    )
    assert out["mean_infected"] == 3.0
    assert out["mean_total_harm"] == 7.5
    assert out["mean_harm_per_node"] == 2.5


def test_propagation_multiplier_scales_spread() -> None:
    g = _chain_graph()
    out = simulate_ic(
        g,
        [0],
        blocked_nodes=[],
        p=1.0,
        n_runs=1,
        random_seed=9,
        propagation_multiplier=0.0,
    )
    assert out["mean_infected"] == 1.0


def main() -> None:
    test_chain_p1_reaches_all()
    test_chain_blocked_middle_stops()
    test_empty_seeds_zero_infected()
    test_p0_only_seeds()
    test_blocked_seed_removed()
    test_reproducibility()
    test_communities_key_present()
    test_harm_metrics_present()
    test_propagation_multiplier_scales_spread()
    print("test_diffusion: all checks passed.")


if __name__ == "__main__":
    main()
