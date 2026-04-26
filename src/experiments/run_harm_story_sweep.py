"""
Targeted sweep for the final harm-aware project story.

This is narrower than ``run_harm_matrix.py`` and is designed to quickly surface
the strongest report-ready settings:

- larger egos
- stronger severity
- budgets where method differences are visible
- both constant and severity-scaled propagation
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
from src.experiments.ego_loader import load_ego_from_manifest_entry, load_manifest, validate_seeds_and_block
from src.experiments.harm_common import compute_harm_method_blocked_sets, load_severity_specs, method_objective_family
from src.experiments.seed_utils import build_seeds
from src.models.diffusion import simulate_ic
from src.models.interventions import build_default_greedy_candidate_pool, sanitize_budget

logger = logging.getLogger(__name__)

N_RUNS = 32
GREEDY_POOL_TOP_N = 15
EVAL_SEED = 818_181
FAIRNESS_LAMBDA = 3.0
RESILIENCE_LAMBDA = 3.0

PROPAGATION_P = {"medium": 0.03, "high": 0.05}
PROPAGATION_MODES = ["constant", "bucket"]
SEED_STRATEGIES = ["high_degree", "community_concentrated"]
SEVERITY_REGIMES = ["medium", "high"]
BUDGETS = [10, 20]
N_SEEDS = 10
EGO_LABELS = ["moderate_large_circles", "large_circles"]


def _cell_integers(
    ego_id: int,
    seed_strategy: str,
    propagation_regime: str,
    budget: int,
    severity_regime: str,
    propagation_mode: str,
) -> Tuple[int, int, int]:
    raw = (
        f"{ego_id}|{seed_strategy}|{propagation_regime}|{budget}|"
        f"{severity_regime}|{propagation_mode}|harm_story_sweep"
    ).encode()
    h = int(hashlib.sha256(raw).hexdigest()[:16], 16)
    seed_sel = 1_000_000 + (h % 8_999_999)
    rnd_blk = 10_000_000 + ((h // 9_000_000) % 8_999_999)
    grd = 20_000_000 + ((h // 81_000_000_000_000) % 8_999_999)
    return seed_sel, rnd_blk, grd


def _safe_build_seeds(
    graph,
    strategy: str,
    communities: Dict[int, int],
    rng: np.random.Generator,
) -> List[int]:
    try:
        return build_seeds(graph, strategy, N_SEEDS, communities, rng)
    except (RuntimeError, ValueError) as exc:
        logger.warning("Falling back to high_degree seeds (%s)", exc)
        return build_seeds(graph, "high_degree", N_SEEDS, communities, rng)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = load_manifest()
    by_label = {str(entry["label"]): entry for entry in manifest["egos"]}

    out_csv = OUTPUTS / "tables" / "harm_story_sweep_results.csv"
    out_log = OUTPUTS / "logs" / "harm_story_sweep_run.txt"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_log.parent.mkdir(parents=True, exist_ok=True)
    out_csv.unlink(missing_ok=True)

    header_written = False
    all_rows: List[dict] = []
    log_lines: List[str] = []
    cell_idx = 0

    for propagation_mode in PROPAGATION_MODES:
        severity_specs = load_severity_specs(propagation_mode)
        for ego_label in EGO_LABELS:
            bundle = load_ego_from_manifest_entry(by_label[ego_label])
            for seed_strategy in SEED_STRATEGIES:
                for propagation_regime, p in PROPAGATION_P.items():
                    for budget_requested in BUDGETS:
                        for severity_regime in SEVERITY_REGIMES:
                            spec = severity_specs[severity_regime]
                            seed_sel, rnd_blk, grd = _cell_integers(
                                bundle.ego_id,
                                seed_strategy,
                                propagation_regime,
                                budget_requested,
                                severity_regime,
                                propagation_mode,
                            )
                            rng = np.random.default_rng(seed_sel)
                            seeds = _safe_build_seeds(bundle.graph, seed_strategy, bundle.communities, rng)
                            budget_sanitized = sanitize_budget(bundle.graph, budget_requested, exclude=seeds)
                            greedy_pool = build_default_greedy_candidate_pool(
                                bundle.graph,
                                seeds,
                                top_n=GREEDY_POOL_TOP_N,
                            )
                            blocked_by_method = compute_harm_method_blocked_sets(
                                bundle.graph,
                                seeds,
                                budget_sanitized,
                                p,
                                bundle.communities,
                                n_runs_greedy=N_RUNS,
                                random_block_seed=rnd_blk,
                                greedy_seed=grd,
                                fairness_lambda=FAIRNESS_LAMBDA,
                                resilience_lambda=RESILIENCE_LAMBDA,
                                pool_top_n=GREEDY_POOL_TOP_N,
                                severity_profile=spec.profile,
                            )
                            log_lines.append(
                                (
                                    f"ego={bundle.label} seed={seed_strategy} p_regime={propagation_regime} "
                                    f"budget={budget_requested} severity={severity_regime} prop_mode={propagation_mode} "
                                    f"score={spec.profile.severity_score:.3f} harm_w={spec.profile.base_harm_weight:.2f} "
                                    f"prop_mult={spec.profile.propagation_multiplier:.3f} greedy_pool={len(greedy_pool)}"
                                )
                            )
                            for method, blocked in blocked_by_method.items():
                                validate_seeds_and_block(seeds, blocked, bundle.graph)
                                t0 = time.perf_counter()
                                res = simulate_ic(
                                    bundle.graph,
                                    seeds,
                                    blocked_nodes=blocked,
                                    p=p,
                                    n_runs=N_RUNS,
                                    communities=bundle.communities,
                                    random_seed=EVAL_SEED,
                                    scenario_harm_weight=spec.profile.base_harm_weight,
                                    propagation_multiplier=spec.profile.propagation_multiplier,
                                )
                                elapsed = time.perf_counter() - t0
                                row = {
                                    "ego_id": bundle.ego_id,
                                    "ego_label": bundle.label,
                                    "community_source": bundle.community_source,
                                    "seed_strategy": seed_strategy,
                                    "propagation_regime": propagation_regime,
                                    "p": p,
                                    "propagation_mode": propagation_mode,
                                    "propagation_multiplier": spec.profile.propagation_multiplier,
                                    "severity_regime": severity_regime,
                                    "severity_score": spec.profile.severity_score,
                                    "base_harm_weight": spec.profile.base_harm_weight,
                                    "budget_requested": budget_requested,
                                    "budget_sanitized": budget_sanitized,
                                    "budget_k": len(blocked),
                                    "greedy_pool_size": len(greedy_pool),
                                    "pool_may_limit_greedy_family": len(greedy_pool) < budget_sanitized,
                                    "method": method,
                                    "method_objective_family": method_objective_family(method),
                                    "mean_infected": res["mean_infected"],
                                    "std_infected": res["std_infected"],
                                    "mean_infected_stderr": res["std_infected"] / float(np.sqrt(N_RUNS)),
                                    "mean_total_harm": res["mean_total_harm"],
                                    "std_total_harm": res["std_total_harm"],
                                    "mean_total_harm_stderr": res["std_total_harm"] / float(np.sqrt(N_RUNS)),
                                    "mean_harm_per_node": res["mean_harm_per_node"],
                                    "mean_worst_community_infection_rate": res["mean_worst_community_infection_rate"],
                                    "mean_worst_community_harm_rate": res["mean_worst_community_harm_rate"],
                                    "runtime_seconds": round(elapsed, 5),
                                    "n_runs_eval": N_RUNS,
                                    "n_runs_greedy_oracle": N_RUNS,
                                    "blocked_nodes_json": json.dumps(sorted(blocked)),
                                    "seeds_json": json.dumps(sorted(seeds)),
                                }
                                all_rows.append(row)
                                pd.DataFrame([row]).to_csv(
                                    out_csv,
                                    mode="w" if not header_written else "a",
                                    header=not header_written,
                                    index=False,
                                )
                                header_written = True
                            cell_idx += 1
                            if cell_idx % 10 == 0:
                                logger.info("Checkpoint: finished %d story cells", cell_idx)

    header = (
        "Targeted harm story sweep\n"
        f"cells={cell_idx} rows={len(all_rows)} n_runs={N_RUNS}\n"
        "Purpose: final-report slice search for larger egos and stronger severity settings.\n\n"
    )
    out_log.write_text(header + "\n".join(log_lines), encoding="utf-8")
    logger.info("Wrote %s (%d rows)", out_csv, len(all_rows))


if __name__ == "__main__":
    main()
