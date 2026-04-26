"""
Focused binary-vs-harm-aware method comparison.

This runner is intentionally narrower than the full matrix so it is practical to
reproduce during final-project polishing while still exercising the new severity
and harm-aware objectives end to end.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import OUTPUTS
from src.experiments.ego_loader import (
    EgoBundle,
    load_ego_from_manifest_entry,
    load_manifest,
    validate_seeds_and_block,
)
from src.experiments.harm_common import (
    compute_harm_method_blocked_sets,
    load_severity_specs,
    method_objective_family,
)
from src.experiments.seed_utils import build_seeds
from src.models.diffusion import simulate_ic
from src.models.interventions import build_default_greedy_candidate_pool, sanitize_budget

logger = logging.getLogger(__name__)

N_RUNS_COMPARISON = 32
GREEDY_POOL_TOP_N = 15
EVAL_SEED = 404_041
FAIRNESS_LAMBDA = 3.0
RESILIENCE_LAMBDA = 3.0
P_MEDIUM = 0.03
SEED_STRATEGY = "high_degree"
N_SEEDS = 10
BUDGETS_REQUESTED = [5, 10, 20]
EGO_LABELS = [
    "medium_circles",
    "moderate_large_circles",
    "detected_communities",
    "large_circles",
]
SEVERITY_REGIMES = ["low", "medium", "high"]
PROPAGATION_MODE = "constant"


def _cell_integers(
    ego_id: int,
    severity_regime: str,
    budget_requested: int,
) -> Tuple[int, int, int]:
    raw = f"{ego_id}|{severity_regime}|{budget_requested}|harm_method_comparison".encode()
    h = int(hashlib.sha256(raw).hexdigest()[:16], 16)
    seed_sel = 1_000_000 + (h % 8_999_999)
    rnd_blk = 10_000_000 + ((h // 9_000_000) % 8_999_999)
    grd = 20_000_000 + ((h // 81_000_000_000_000) % 8_999_999)
    return seed_sel, rnd_blk, grd


def _safe_build_seeds(
    graph,
    strategy: str,
    n_seeds: int,
    communities: Dict[int, int],
    rng: np.random.Generator,
) -> List[int]:
    try:
        return build_seeds(graph, strategy, n_seeds, communities, rng)
    except (RuntimeError, ValueError) as exc:
        logger.warning("Falling back to high_degree seeds (%s)", exc)
        return build_seeds(graph, "high_degree", n_seeds, communities, rng)


def run_one_setting(
    bundle: EgoBundle,
    severity_regime: str,
    budget_requested: int,
) -> Tuple[List[dict], List[str]]:
    specs = load_severity_specs(PROPAGATION_MODE)
    spec = specs[severity_regime]
    g = bundle.graph
    comm = bundle.communities
    seed_sel, rnd_blk, grd = _cell_integers(bundle.ego_id, severity_regime, budget_requested)
    rng = np.random.default_rng(seed_sel)
    seeds = _safe_build_seeds(g, SEED_STRATEGY, N_SEEDS, comm, rng)
    budget_sanitized = sanitize_budget(g, budget_requested, exclude=seeds)
    greedy_pool = build_default_greedy_candidate_pool(g, seeds, top_n=GREEDY_POOL_TOP_N)
    max_blockers_graph = max(0, g.number_of_nodes() - len(seeds))

    blocked_by_method = compute_harm_method_blocked_sets(
        g,
        seeds,
        budget_sanitized,
        P_MEDIUM,
        comm,
        n_runs_greedy=N_RUNS_COMPARISON,
        random_block_seed=rnd_blk,
        greedy_seed=grd,
        fairness_lambda=FAIRNESS_LAMBDA,
        resilience_lambda=RESILIENCE_LAMBDA,
        pool_top_n=GREEDY_POOL_TOP_N,
        severity_profile=spec.profile,
    )
    log_lines = [
        (
            f"ego={bundle.ego_id} label={bundle.label} severity={severity_regime} "
            f"score={spec.profile.severity_score:.3f} harm_w={spec.profile.base_harm_weight:.2f} "
            f"prop_mode={PROPAGATION_MODE} prop_mult={spec.profile.propagation_multiplier:.3f} "
            f"budget_req={budget_requested} budget_sanitized={budget_sanitized} "
            f"greedy_pool_size={len(greedy_pool)} n_runs={N_RUNS_COMPARISON}"
        ),
        f"seeds={sorted(seeds)}",
    ]

    rows: List[dict] = []
    for method, blocked in blocked_by_method.items():
        validate_seeds_and_block(seeds, blocked, g)
        t0 = time.perf_counter()
        res = simulate_ic(
            g,
            seeds,
            blocked_nodes=blocked,
            p=P_MEDIUM,
            n_runs=N_RUNS_COMPARISON,
            communities=comm,
            random_seed=EVAL_SEED,
            scenario_harm_weight=spec.profile.base_harm_weight,
            propagation_multiplier=spec.profile.propagation_multiplier,
        )
        elapsed = time.perf_counter() - t0
        rows.append(
            {
                "ego_id": bundle.ego_id,
                "ego_label": bundle.label,
                "matrix_role": bundle.matrix_role,
                "community_source": bundle.community_source,
                "seed_strategy": SEED_STRATEGY,
                "propagation_regime": "medium",
                "p": P_MEDIUM,
                "severity_regime": severity_regime,
                "severity_score": spec.profile.severity_score,
                "severity_label": spec.profile.severity_label,
                "base_harm_weight": spec.profile.base_harm_weight,
                "propagation_mode": PROPAGATION_MODE,
                "propagation_multiplier": spec.profile.propagation_multiplier,
                "budget_requested": budget_requested,
                "budget_sanitized": budget_sanitized,
                "budget_k": len(blocked),
                "max_blockers_graph": max_blockers_graph,
                "greedy_pool_size": len(greedy_pool),
                "pool_may_limit_greedy_family": len(greedy_pool) < budget_sanitized,
                "method": method,
                "method_objective_family": method_objective_family(method),
                "mean_infected": res["mean_infected"],
                "std_infected": res["std_infected"],
                "mean_infected_stderr": res["std_infected"] / float(np.sqrt(N_RUNS_COMPARISON)),
                "mean_total_harm": res["mean_total_harm"],
                "std_total_harm": res["std_total_harm"],
                "mean_total_harm_stderr": res["std_total_harm"] / float(np.sqrt(N_RUNS_COMPARISON)),
                "mean_harm_per_node": res["mean_harm_per_node"],
                "mean_infection_rate": res["mean_infection_rate"],
                "mean_worst_community_infection_rate": res["mean_worst_community_infection_rate"],
                "mean_worst_community_harm_rate": res["mean_worst_community_harm_rate"],
                "runtime_seconds": round(elapsed, 5),
                "n_runs_eval": N_RUNS_COMPARISON,
                "n_runs_greedy_oracle": N_RUNS_COMPARISON,
                "blocked_nodes_json": json.dumps(sorted(blocked)),
                "seeds_json": json.dumps(sorted(seeds)),
            }
        )
        log_lines.append(f"  {method}: k={len(blocked)} blocked={sorted(blocked)}")
    return rows, log_lines


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = load_manifest()
    by_label = {str(entry["label"]): entry for entry in manifest["egos"]}

    out_csv = OUTPUTS / "tables" / "harm_method_comparison_results.csv"
    out_log = OUTPUTS / "logs" / "harm_method_comparison_run.txt"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_log.parent.mkdir(parents=True, exist_ok=True)
    out_csv.unlink(missing_ok=True)

    header_written = False
    all_logs: List[str] = []
    all_rows: List[dict] = []

    for label in EGO_LABELS:
        bundle = load_ego_from_manifest_entry(by_label[label])
        for severity_regime in SEVERITY_REGIMES:
            for budget_requested in BUDGETS_REQUESTED:
                rows, logs = run_one_setting(bundle, severity_regime, budget_requested)
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

    header = (
        "Harm-aware method comparison\n"
        f"egos={len(EGO_LABELS)} severity_regimes={len(SEVERITY_REGIMES)} budgets={len(BUDGETS_REQUESTED)} n_runs={N_RUNS_COMPARISON}\n"
        "Structural communities only; no demographic interpretation.\n"
        "Semi-synthetic severity scenarios; no user-level CoAID-to-SNAP mapping.\n\n"
    )
    out_log.write_text(header + "\n".join(all_logs), encoding="utf-8")
    logger.info("Wrote %s (%d rows)", out_csv, len(all_rows))


if __name__ == "__main__":
    main()
