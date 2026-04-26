"""
Smoke experiment: IC diffusion with simple blocking baselines on the dev ego graph.

Simulation on real **graph structure** only; not a causal claim about Twitter contagion.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Set

import networkx as nx
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, OUTPUTS
from src.experiments.seed_utils import build_seeds
from src.models.diffusion import simulate_ic
from src.models.interventions import (
    select_no_intervention,
    select_random_blocking,
    select_top_degree_blocking,
)
from src.models.interventions import sanitize_budget

logger = logging.getLogger(__name__)

SMOKE_N_RUNS = 200
SMOKE_RANDOM_SEED = 42


def load_dev_graph() -> nx.Graph:
    path = DATA_PROCESSED / "twitter_dev_edges.csv"
    df = pd.read_csv(path)
    return nx.from_pandas_edgelist(df, "u", "v")


def load_community_map() -> Dict[int, int]:
    path = DATA_PROCESSED / "twitter_dev_node_communities.csv"
    df = pd.read_csv(path)
    out: Dict[int, int] = {}
    for _, row in df.iterrows():
        out[int(row["node_id"])] = int(row["primary_circle_id"])
    return out


def load_scenario(name: str = "medium_virality_high_degree.json") -> dict:
    p = DATA_PROCESSED / "scenarios" / name
    return json.loads(p.read_text(encoding="utf-8"))


def run_smoke() -> tuple[pd.DataFrame, str]:
    graph = load_dev_graph()
    communities = load_community_map()
    scenario = load_scenario()
    meta = json.loads((DATA_PROCESSED / "twitter_dev_metadata.json").read_text(encoding="utf-8"))

    rng = np.random.default_rng(SMOKE_RANDOM_SEED)
    seeds = build_seeds(
        graph,
        scenario["seed_strategy"],
        int(scenario["n_seeds"]),
        communities,
        rng,
    )
    seed_set = set(seeds)

    budgets = scenario["intervention_budgets"]
    k = int(budgets[0])
    k = sanitize_budget(graph, k, exclude=seeds)

    p = float(scenario["ic_propagation_p"])
    n_runs = SMOKE_N_RUNS

    rows: List[dict] = []

    def record(method: str, blocked: Set[int]) -> None:
        res = simulate_ic(
            graph,
            seeds,
            blocked_nodes=blocked,
            p=p,
            n_runs=n_runs,
            communities=communities,
            random_seed=SMOKE_RANDOM_SEED,
        )
        rows.append(
            {
                "method": method,
                "budget_k": len(blocked),
                "n_runs": n_runs,
                "ic_p": p,
                "n_seeds": len(seeds),
                "seed_strategy": scenario["seed_strategy"],
                "ego_id": meta.get("ego_id"),
                "mean_infected": res["mean_infected"],
                "std_infected": res["std_infected"],
                "mean_infection_rate": res["mean_infection_rate"],
                "mean_steps": res["mean_steps"],
            }
        )

    record("no_intervention", select_no_intervention(graph, k))
    record(
        "random_blocking",
        select_random_blocking(graph, k, random_seed=SMOKE_RANDOM_SEED + 1, exclude=seeds),
    )
    record("top_degree_blocking", select_top_degree_blocking(graph, k, exclude=seeds))

    df = pd.DataFrame(rows)
    summary_lines = [
        "Smoke baselines (simulation study; not causal real-world spread)",
        f"ego_id={meta.get('ego_id')} n_nodes={graph.number_of_nodes()} n_edges={graph.number_of_edges()}",
        f"seeds={seeds} strategy={scenario['seed_strategy']} p={p} n_runs={n_runs} budget_k={k}",
        "",
        df.to_string(index=False),
        "",
        "Note: structural communities (circles) are network-defined, not demographic groups.",
    ]
    if graph.number_of_nodes() < 200:
        summary_lines += [
            "",
            "Caution: dev graph is small; baseline differences may be modest; consider a larger ego for main experiments.",
        ]
    summary = "\n".join(summary_lines)
    return df, summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out_dir = OUTPUTS / "tables"
    log_dir = OUTPUTS / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    df, summary = run_smoke()
    csv_path = out_dir / "smoke_baseline_results.csv"
    log_path = log_dir / "smoke_run_summary.txt"
    df.to_csv(csv_path, index=False)
    log_path.write_text(summary, encoding="utf-8")
    logger.info("Wrote %s", csv_path)
    logger.info("Wrote %s", log_path)
    print(summary)


if __name__ == "__main__":
    main()
