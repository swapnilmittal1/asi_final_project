"""
Compare blocking interventions on fixed seeds (simulation study).

Aligned greedy oracle / evaluation (same ``n_runs``) for fair comparison with the full matrix.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, OUTPUTS, SCENARIOS_DIR
from src.experiments.ego_loader import (
    load_ego_from_manifest_entry,
    load_manifest,
    validate_seeds_and_block,
)
from src.experiments.matrix_common import compute_method_blocked_sets
from src.experiments.seed_utils import build_seeds
from src.models.diffusion import simulate_ic
from src.models.interventions import select_random_blocking, sanitize_budget

logger = logging.getLogger(__name__)

N_RUNS_UNIFIED = 64
EVAL_RANDOM_SEED = 123_456
RANDOM_BLOCK_SEED = 9_001
GREEDY_ORACLE_SEED = 77_777
FAIRNESS_LAMBDA = 3.0
GREEDY_POOL_TOP_N = 25


def load_scenario(name: str) -> dict:
    path = SCENARIOS_DIR / name
    if not path.suffix:
        path = path.with_suffix(".json")
    return json.loads(path.read_text(encoding="utf-8"))


def run_job(
    bundle,
    scenario: dict,
    scenario_name: str,
    budget_k: int,
) -> Tuple[List[dict], List[str]]:
    g = bundle.graph
    comm = bundle.communities
    rng = np.random.default_rng(42)
    seeds = build_seeds(
        g,
        str(scenario["seed_strategy"]),
        int(scenario["n_seeds"]),
        comm,
        rng,
    )
    p = float(scenario["ic_propagation_p"])
    budget = sanitize_budget(g, int(budget_k), exclude=seeds)

    log_lines = [
        f"ego_id={bundle.ego_id} label={bundle.label} scenario={scenario_name} budget={budget} "
        f"n_seeds={len(seeds)} p={p} n_runs_unified={N_RUNS_UNIFIED}"
    ]
    log_lines.append(f"seed_set={sorted(seeds)}")
    log_lines.append(f"community_source={bundle.community_source}")

    blocked_by_method = compute_method_blocked_sets(
        g,
        seeds,
        budget,
        p,
        comm,
        n_runs_greedy=N_RUNS_UNIFIED,
        random_block_seed=RANDOM_BLOCK_SEED,
        greedy_seed=GREEDY_ORACLE_SEED,
        fairness_lambda=FAIRNESS_LAMBDA,
        pool_top_n=GREEDY_POOL_TOP_N,
    )

    rows: List[dict] = []
    for method, blocked in blocked_by_method.items():
        validate_seeds_and_block(seeds, blocked, g)
        t0 = time.perf_counter()
        res = simulate_ic(
            g,
            seeds,
            blocked_nodes=blocked,
            p=p,
            n_runs=N_RUNS_UNIFIED,
            communities=comm,
            random_seed=EVAL_RANDOM_SEED,
        )
        elapsed = time.perf_counter() - t0
        wc = res["mean_worst_community_infection_rate"]
        rows.append(
            {
                "ego_id": bundle.ego_id,
                "ego_label": bundle.label,
                "scenario_name": scenario_name,
                "method": method,
                "budget_k": len(blocked),
                "mean_infected": res["mean_infected"],
                "std_infected": res["std_infected"],
                "mean_infection_rate": res["mean_infection_rate"],
                "mean_steps": res["mean_steps"],
                "worst_community_infection_rate": wc if wc is not None else float("nan"),
                "community_source": bundle.community_source,
                "runtime_seconds": round(elapsed, 4),
                "blocked_nodes_json": json.dumps(sorted(blocked)),
                "n_seeds": len(seeds),
                "ic_p": p,
                "n_report_runs": N_RUNS_UNIFIED,
            }
        )
        log_lines.append(f"{method}: blocked={sorted(blocked)}")

    b1 = select_random_blocking(g, budget, random_seed=RANDOM_BLOCK_SEED, exclude=set(seeds))
    b2 = select_random_blocking(g, budget, random_seed=RANDOM_BLOCK_SEED, exclude=set(seeds))
    if b1 != b2:
        raise RuntimeError("random_blocking not reproducible with fixed seed")

    return rows, log_lines


def default_validation_jobs():
    return [
        ("dev", "medium_virality_high_degree", 10),
        ("medium_circles", "medium_virality_high_degree", 10),
        ("dev", "medium_virality_high_degree", 5),
        ("dev", "low_virality_random", 10),
        ("detected_communities", "medium_virality_high_degree", 10),
        ("moderate_large_circles", "medium_virality_high_degree", 10),
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = load_manifest()
    by_label = {str(e["label"]): e for e in manifest["egos"]}

    all_rows: List[dict] = []
    all_logs: List[str] = []

    for key, scen, bud in default_validation_jobs():
        entry = by_label[key]
        bundle = load_ego_from_manifest_entry(entry)
        scenario = load_scenario(scen)
        rows, logs = run_job(bundle, scenario, scen, bud)
        all_rows.extend(rows)
        all_logs.extend(logs)
        all_logs.append("")

    out_csv = OUTPUTS / "tables" / "intervention_comparison_validation.csv"
    out_log = OUTPUTS / "logs" / "intervention_comparison_validation.txt"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_log.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    header = (
        "Intervention comparison validation\n"
        "Simulation on observed graph topology only; not causal inference.\n"
        "Structural communities are not demographic groups.\n"
        f"Greedy oracle n_runs = eval n_runs = {N_RUNS_UNIFIED}\n\n"
    )
    out_log.write_text(header + "\n".join(all_logs), encoding="utf-8")
    logger.info("Wrote %s (%d rows)", out_csv, len(all_rows))


if __name__ == "__main__":
    main()
