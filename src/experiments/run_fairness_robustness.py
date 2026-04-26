"""
Fairness-aware greedy: sweep ``lambda_fair`` (structural communities only).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import OUTPUTS
from src.experiments.ego_loader import load_ego_from_manifest_entry, load_manifest
from src.experiments.seed_utils import build_seeds
from src.models.diffusion import simulate_ic
from src.models.interventions import (
    build_default_greedy_candidate_pool,
    select_fairness_aware_greedy_blocking,
    select_greedy_blocking,
    sanitize_budget,
)

logger = logging.getLogger(__name__)

N_RUNS = 48
EVAL_SEED = 927_172
P_MEDIUM = 0.03
N_SEEDS = 10
GREEDY_POOL_TOP_N = 15
LAMBDAS = [0.5, 1.0, 3.0, 5.0]
BUDGETS = [10, 20]
EGO_LABELS = ["moderate_large_circles", "large_circles"]


def _ego_seeds_rng(ego_id: int) -> int:
    h = int(hashlib.sha256(f"fairness_seeds|{ego_id}".encode()).hexdigest()[:12], 16)
    return 3_000_000 + (h % 8_000_000)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = load_manifest()
    by_label = {str(e["label"]): e for e in manifest["egos"]}
    rows: List[dict] = []
    logs: List[str] = []

    for label in EGO_LABELS:
        bundle = load_ego_from_manifest_entry(by_label[label])
        g = bundle.graph
        comm = bundle.communities
        rng = np.random.default_rng(_ego_seeds_rng(bundle.ego_id))
        seeds = build_seeds(g, "high_degree", N_SEEDS, comm, rng)
        logs.append(f"ego {bundle.ego_id} {label} seeds={sorted(seeds)}")

        for budget_req in BUDGETS:
            budget = sanitize_budget(g, budget_req, exclude=seeds)
            max_bg = max(0, g.number_of_nodes() - len(seeds))
            pool_sz = len(
                build_default_greedy_candidate_pool(g, seeds, top_n=GREEDY_POOL_TOP_N)
            )
            gr_seed = 40_000_000 + bundle.ego_id % 1_000_000 + budget_req * 97

            t0 = time.perf_counter()
            b_g = select_greedy_blocking(
                g,
                seeds,
                budget,
                p=P_MEDIUM,
                n_runs=N_RUNS,
                communities=comm,
                random_seed=gr_seed,
                candidate_pool_top_n=GREEDY_POOL_TOP_N,
            )
            t_g = time.perf_counter() - t0
            res_g = simulate_ic(
                g,
                seeds,
                blocked_nodes=b_g,
                p=P_MEDIUM,
                n_runs=N_RUNS,
                communities=comm,
                random_seed=EVAL_SEED,
            )
            si = float(res_g["std_infected"])
            rows.append(
                {
                    "ego_id": bundle.ego_id,
                    "ego_label": label,
                    "community_source": bundle.community_source,
                    "budget_requested": budget_req,
                    "budget_sanitized": budget,
                    "max_blockers_graph": max_bg,
                    "greedy_pool_size": pool_sz,
                    "budget_k": len(b_g),
                    "lambda_fair": float("nan"),
                    "method": "greedy_blocking_reference",
                    "mean_infected": res_g["mean_infected"],
                    "std_infected": si,
                    "mean_infected_stderr": si / (N_RUNS**0.5) if N_RUNS > 1 else 0.0,
                    "mean_worst_community_infection_rate": res_g["mean_worst_community_infection_rate"],
                    "runtime_selection_seconds": round(t_g, 4),
                    "n_runs_oracle_and_eval": N_RUNS,
                    "blocked_nodes_json": json.dumps(sorted(b_g)),
                }
            )

            for lam in LAMBDAS:
                t1 = time.perf_counter()
                b_f = select_fairness_aware_greedy_blocking(
                    g,
                    seeds,
                    budget,
                    communities=comm,
                    lambda_fair=float(lam),
                    p=P_MEDIUM,
                    n_runs=N_RUNS,
                    random_seed=gr_seed + int(lam * 1000) + 17,
                    candidate_pool_top_n=GREEDY_POOL_TOP_N,
                )
                t_f = time.perf_counter() - t1
                res_f = simulate_ic(
                    g,
                    seeds,
                    blocked_nodes=b_f,
                    p=P_MEDIUM,
                    n_runs=N_RUNS,
                    communities=comm,
                    random_seed=EVAL_SEED,
                )
                sif = float(res_f["std_infected"])
                rows.append(
                    {
                        "ego_id": bundle.ego_id,
                        "ego_label": label,
                        "community_source": bundle.community_source,
                        "budget_requested": budget_req,
                        "budget_sanitized": budget,
                        "max_blockers_graph": max_bg,
                        "greedy_pool_size": pool_sz,
                        "budget_k": len(b_f),
                        "lambda_fair": lam,
                        "method": "fairness_aware_greedy_blocking",
                        "mean_infected": res_f["mean_infected"],
                        "std_infected": sif,
                        "mean_infected_stderr": sif / (N_RUNS**0.5) if N_RUNS > 1 else 0.0,
                        "mean_worst_community_infection_rate": res_f["mean_worst_community_infection_rate"],
                        "runtime_selection_seconds": round(t_f, 4),
                        "n_runs_oracle_and_eval": N_RUNS,
                        "blocked_nodes_json": json.dumps(sorted(b_f)),
                    }
                )
                logs.append(
                    f"  budget={budget_req} lambda={lam} mean_I={res_f['mean_infected']:.3f} "
                    f"worst={res_f['mean_worst_community_infection_rate']:.3f}"
                )

    out_csv = OUTPUTS / "tables" / "fairness_robustness_results.csv"
    out_log = OUTPUTS / "logs" / "fairness_robustness_run.txt"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_log.write_text(
        "Fairness lambda sweep (oracle n_runs = eval n_runs)\n\n" + "\n".join(logs),
        encoding="utf-8",
    )
    logger.info("Wrote %s (%d rows)", out_csv, len(rows))


if __name__ == "__main__":
    main()
