"""
Full factorial experiment matrix (simulation study).

Greedy oracle ``n_runs`` equals final evaluation ``n_runs`` for aligned comparisons.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from src.config import OUTPUTS
from src.experiments.ego_loader import (
    EgoBundle,
    load_ego_from_manifest_entry,
    load_manifest,
    validate_seeds_and_block,
)
from src.experiments.matrix_common import compute_method_blocked_sets
from src.experiments.seed_utils import build_seeds
from src.models.diffusion import simulate_ic
from src.models.interventions import build_default_greedy_candidate_pool, sanitize_budget

logger = logging.getLogger(__name__)

# Aligned IC budget: oracle == final report (see docs/diffusion_notes.md)
N_RUNS_MATRIX = 48
GREEDY_POOL_TOP_N = 15
MATRIX_EVAL_SEED = 314_159_265
FAIRNESS_LAMBDA_MATRIX = 3.0

PROPAGATION_P = {"low": 0.01, "medium": 0.03, "high": 0.05}
SEED_STRATEGIES = ["random", "high_degree", "community_concentrated"]
BUDGETS_REQUESTED = [5, 10, 20]
N_SEEDS_MATRIX = 10

# Order: primary egos first, dev last (debugging-scale)
MATRIX_EGO_LABELS = [
    "medium_circles",
    "moderate_large_circles",
    "detected_communities",
    "large_circles",
    "dev",
]


def _cell_integers(ego_id: int, seed_strategy: str, regime: str, budget: int) -> Tuple[int, int, int]:
    raw = f"{ego_id}|{seed_strategy}|{regime}|{budget}".encode()
    h = int(hashlib.sha256(raw).hexdigest()[:16], 16)
    seed_sel = 1_000_000 + (h % 8_999_999)
    rnd_blk = 10_000_000 + ((h // 9_000_000) % 8_999_999)
    grd = 20_000_000 + ((h // 81_000_000_000_000) % 8_999_999)
    return seed_sel, rnd_blk, grd


def _safe_build_seeds(
    graph: Any,
    strategy: str,
    n_seeds: int,
    communities: Dict[int, int],
    rng: np.random.Generator,
) -> List[int]:
    try:
        return build_seeds(graph, strategy, n_seeds, communities, rng)
    except (RuntimeError, ValueError) as e:
        logger.warning("Falling back to high_degree seeds (%s)", e)
        return build_seeds(graph, "high_degree", n_seeds, communities, rng)


def run_one_cell(
    bundle: EgoBundle,
    seed_strategy: str,
    regime: str,
    budget_req: int,
) -> Tuple[List[dict], List[str]]:
    g = bundle.graph
    comm = bundle.communities
    p = float(PROPAGATION_P[regime])
    seed_sel, rnd_blk, grd = _cell_integers(bundle.ego_id, seed_strategy, regime, budget_req)

    rng = np.random.default_rng(seed_sel)
    seeds = _safe_build_seeds(g, seed_strategy, N_SEEDS_MATRIX, comm, rng)
    budget_sanitized = sanitize_budget(g, int(budget_req), exclude=seeds)
    n_nodes = g.number_of_nodes()
    max_blockers_graph = max(0, n_nodes - len(seeds))
    greedy_pool = build_default_greedy_candidate_pool(
        g,
        seeds,
        top_n=GREEDY_POOL_TOP_N,
    )
    greedy_pool_size = len(greedy_pool)
    pool_limits_greedy_family = greedy_pool_size < budget_sanitized

    scenario_name = f"matrix_{regime}_{seed_strategy}_b{budget_req}"

    log_lines = [
        f"cell ego={bundle.ego_id} label={bundle.label} role={bundle.matrix_role} "
        f"seed_strat={seed_strategy} regime={regime} p={p} budget_req={budget_req} "
        f"budget_sanitized={budget_sanitized} max_blockers_graph={max_blockers_graph} "
        f"greedy_pool_size={greedy_pool_size} pool_limits_greedy={pool_limits_greedy_family} "
        f"n_runs={N_RUNS_MATRIX} eval_seed={MATRIX_EVAL_SEED} seed_sel_seed={seed_sel}",
        f"seeds={sorted(seeds)}",
    ]

    budget = budget_sanitized

    blocked_by_method = compute_method_blocked_sets(
        g,
        seeds,
        budget,
        p,
        comm,
        n_runs_greedy=N_RUNS_MATRIX,
        random_block_seed=rnd_blk,
        greedy_seed=grd,
        fairness_lambda=FAIRNESS_LAMBDA_MATRIX,
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
            n_runs=N_RUNS_MATRIX,
            communities=comm,
            random_seed=MATRIX_EVAL_SEED,
        )
        elapsed = time.perf_counter() - t0
        wc = res["mean_worst_community_infection_rate"]
        std_i = float(res["std_infected"])
        se_mean = std_i / (float(N_RUNS_MATRIX) ** 0.5) if N_RUNS_MATRIX > 1 else 0.0
        rows.append(
            {
                "ego_id": bundle.ego_id,
                "ego_label": bundle.label,
                "matrix_role": bundle.matrix_role,
                "community_source": bundle.community_source,
                "scenario_name": scenario_name,
                "seed_strategy": seed_strategy,
                "propagation_regime": regime,
                "budget_k": len(blocked),
                "budget_requested": budget_req,
                "budget_sanitized": budget_sanitized,
                "max_blockers_graph": max_blockers_graph,
                "greedy_pool_size": greedy_pool_size,
                "pool_may_limit_greedy_family": pool_limits_greedy_family,
                "method": method,
                "n_nodes": n_nodes,
                "n_edges": g.number_of_edges(),
                "n_seeds": len(seeds),
                "p": p,
                "mean_infected": res["mean_infected"],
                "std_infected": std_i,
                "mean_infected_stderr": se_mean,
                "mean_infection_rate": res["mean_infection_rate"],
                "mean_steps": res["mean_steps"],
                "mean_worst_community_infection_rate": wc if wc is not None else float("nan"),
                "runtime_seconds": round(elapsed, 5),
                "random_seed_eval": MATRIX_EVAL_SEED,
                "random_seed_seed_selection": seed_sel,
                "random_seed_random_block": rnd_blk,
                "random_seed_greedy": grd,
                "n_runs_greedy_oracle": N_RUNS_MATRIX,
                "n_runs_eval": N_RUNS_MATRIX,
                "blocked_nodes_json": json.dumps(sorted(blocked)),
                "seeds_json": json.dumps(sorted(seeds)),
                "fairness_lambda": FAIRNESS_LAMBDA_MATRIX,
            }
        )
        log_lines.append(f"  {method}: k={len(blocked)} blocked={sorted(blocked)}")

    b1 = blocked_by_method["random_blocking"]
    b2 = compute_method_blocked_sets(
        g,
        seeds,
        budget,
        p,
        comm,
        n_runs_greedy=N_RUNS_MATRIX,
        random_block_seed=rnd_blk,
        greedy_seed=grd,
        fairness_lambda=FAIRNESS_LAMBDA_MATRIX,
        pool_top_n=GREEDY_POOL_TOP_N,
    )["random_blocking"]
    if b1 != b2:
        raise RuntimeError("random_blocking reproducibility failed within cell")

    return rows, log_lines


def reproducibility_spot_check(first_rows: List[dict], bundle: EgoBundle) -> None:
    """Re-run the first stored cell and compare mean_infected per method."""
    if not first_rows:
        return
    r0 = first_rows[0]
    rows2, _ = run_one_cell(
        bundle,
        str(r0["seed_strategy"]),
        str(r0["propagation_regime"]),
        int(r0["budget_requested"]),
    )
    by_m = {r["method"]: r["mean_infected"] for r in rows2}
    for r in first_rows:
        if abs(by_m[r["method"]] - r["mean_infected"]) > 1e-9:
            raise RuntimeError(f"Repro spot-check failed for {r['method']}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = load_manifest()
    by_label = {str(e["label"]): e for e in manifest["egos"]}

    all_rows: List[dict] = []
    all_logs: List[str] = []
    ref_rows: List[dict] | None = None
    ref_bundle: EgoBundle | None = None
    cell_idx = 0

    out_csv = OUTPUTS / "tables" / "full_experiment_results.csv"
    out_log = OUTPUTS / "logs" / "full_experiment_run.txt"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_log.parent.mkdir(parents=True, exist_ok=True)
    out_csv.unlink(missing_ok=True)
    header_written = False

    for label in MATRIX_EGO_LABELS:
        entry = by_label[label]
        bundle = load_ego_from_manifest_entry(entry)
        for seed_strategy in SEED_STRATEGIES:
            for regime in PROPAGATION_P:
                for budget_req in BUDGETS_REQUESTED:
                    rows, logs = run_one_cell(bundle, seed_strategy, regime, budget_req)
                    all_rows.extend(rows)
                    all_logs.extend(logs)
                    all_logs.append("")
                    pd.DataFrame(rows).to_csv(
                        out_csv,
                        mode="w" if not header_written else "a",
                        header=not header_written,
                        index=False,
                    )
                    header_written = True
                    if cell_idx == 0:
                        ref_rows = list(rows)
                        ref_bundle = bundle
                    cell_idx += 1
                    if cell_idx % 10 == 0:
                        logger.info("Checkpoint: finished %d cells", cell_idx)

    if ref_bundle and ref_rows:
        logger.info("Running reproducibility spot-check on first cell...")
        reproducibility_spot_check(ref_rows, ref_bundle)
    n_cells = len(MATRIX_EGO_LABELS) * len(SEED_STRATEGIES) * len(PROPAGATION_P) * len(BUDGETS_REQUESTED)
    header = (
        f"Full experiment matrix\n"
        f"cells={n_cells} rows={len(all_rows)} n_runs_oracle=n_runs_eval={N_RUNS_MATRIX}\n"
        f"Simulation study; structural communities only.\n\n"
    )
    out_log.write_text(header + "\n".join(all_logs), encoding="utf-8")
    logger.info("Wrote %s (%d rows)", out_csv, len(all_rows))


if __name__ == "__main__":
    main()
