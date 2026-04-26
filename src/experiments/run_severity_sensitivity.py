"""
Severity and propagation sensitivity study.
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
from src.experiments.harm_common import compute_harm_method_blocked_sets, load_severity_specs
from src.experiments.seed_utils import build_seeds
from src.models.diffusion import simulate_ic
from src.models.interventions import build_default_greedy_candidate_pool, sanitize_budget

logger = logging.getLogger(__name__)

N_RUNS_SENSITIVITY = 32
GREEDY_POOL_TOP_N = 15
EVAL_SEED = 515_151
FAIRNESS_LAMBDA = 3.0
RESILIENCE_LAMBDA = 3.0
P_MEDIUM = 0.03
SEED_STRATEGY = "high_degree"
N_SEEDS = 10
BUDGET_REQUESTED = 10
EGO_LABELS = ["moderate_large_circles", "large_circles"]
SEVERITY_REGIMES = ["low", "medium", "high"]
PROPAGATION_MODES = ["constant", "bucket", "linear"]
METHODS_TO_KEEP = {
    "greedy_blocking",
    "fairness_aware_greedy_blocking",
    "harm_aware_greedy_blocking",
    "harm_aware_resilience_greedy_blocking",
}


def _cell_integers(ego_id: int, severity_regime: str, propagation_mode: str) -> Tuple[int, int, int]:
    raw = f"{ego_id}|{severity_regime}|{propagation_mode}|severity_sensitivity".encode()
    h = int(hashlib.sha256(raw).hexdigest()[:16], 16)
    seed_sel = 1_000_000 + (h % 8_999_999)
    rnd_blk = 10_000_000 + ((h // 9_000_000) % 8_999_999)
    grd = 20_000_000 + ((h // 81_000_000_000_000) % 8_999_999)
    return seed_sel, rnd_blk, grd


def _safe_build_seeds(graph, communities: Dict[int, int], rng: np.random.Generator) -> List[int]:
    try:
        return build_seeds(graph, SEED_STRATEGY, N_SEEDS, communities, rng)
    except (RuntimeError, ValueError) as exc:
        logger.warning("Falling back to high_degree seeds (%s)", exc)
        return build_seeds(graph, "high_degree", N_SEEDS, communities, rng)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = load_manifest()
    by_label = {str(entry["label"]): entry for entry in manifest["egos"]}

    out_csv = OUTPUTS / "tables" / "severity_sensitivity_results.csv"
    out_log = OUTPUTS / "logs" / "severity_sensitivity_run.txt"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_log.parent.mkdir(parents=True, exist_ok=True)
    out_csv.unlink(missing_ok=True)

    header_written = False
    all_rows: List[dict] = []
    logs: List[str] = []

    for propagation_mode in PROPAGATION_MODES:
        severity_specs = load_severity_specs(propagation_mode)
        for ego_label in EGO_LABELS:
            bundle = load_ego_from_manifest_entry(by_label[ego_label])
            for severity_regime in SEVERITY_REGIMES:
                spec = severity_specs[severity_regime]
                seed_sel, rnd_blk, grd = _cell_integers(bundle.ego_id, severity_regime, propagation_mode)
                rng = np.random.default_rng(seed_sel)
                seeds = _safe_build_seeds(bundle.graph, bundle.communities, rng)
                budget_sanitized = sanitize_budget(bundle.graph, BUDGET_REQUESTED, exclude=seeds)
                greedy_pool = build_default_greedy_candidate_pool(
                    bundle.graph,
                    seeds,
                    top_n=GREEDY_POOL_TOP_N,
                )
                blocked_by_method = compute_harm_method_blocked_sets(
                    bundle.graph,
                    seeds,
                    budget_sanitized,
                    P_MEDIUM,
                    bundle.communities,
                    n_runs_greedy=N_RUNS_SENSITIVITY,
                    random_block_seed=rnd_blk,
                    greedy_seed=grd,
                    fairness_lambda=FAIRNESS_LAMBDA,
                    resilience_lambda=RESILIENCE_LAMBDA,
                    pool_top_n=GREEDY_POOL_TOP_N,
                    severity_profile=spec.profile,
                )
                logs.append(
                    (
                        f"ego={bundle.label} severity={severity_regime} propagation_mode={propagation_mode} "
                        f"score={spec.profile.severity_score:.3f} harm_w={spec.profile.base_harm_weight:.2f} "
                        f"prop_mult={spec.profile.propagation_multiplier:.3f}"
                    )
                )
                for method, blocked in blocked_by_method.items():
                    if method not in METHODS_TO_KEEP:
                        continue
                    validate_seeds_and_block(seeds, blocked, bundle.graph)
                    t0 = time.perf_counter()
                    res = simulate_ic(
                        bundle.graph,
                        seeds,
                        blocked_nodes=blocked,
                        p=P_MEDIUM,
                        n_runs=N_RUNS_SENSITIVITY,
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
                        "seed_strategy": SEED_STRATEGY,
                        "budget_requested": BUDGET_REQUESTED,
                        "budget_sanitized": budget_sanitized,
                        "budget_k": len(blocked),
                        "greedy_pool_size": len(greedy_pool),
                        "p": P_MEDIUM,
                        "propagation_regime": "medium",
                        "propagation_mode": propagation_mode,
                        "propagation_multiplier": spec.profile.propagation_multiplier,
                        "severity_regime": severity_regime,
                        "severity_score": spec.profile.severity_score,
                        "base_harm_weight": spec.profile.base_harm_weight,
                        "method": method,
                        "mean_infected": res["mean_infected"],
                        "std_infected": res["std_infected"],
                        "mean_total_harm": res["mean_total_harm"],
                        "std_total_harm": res["std_total_harm"],
                        "mean_harm_per_node": res["mean_harm_per_node"],
                        "mean_worst_community_infection_rate": res["mean_worst_community_infection_rate"],
                        "mean_worst_community_harm_rate": res["mean_worst_community_harm_rate"],
                        "runtime_seconds": round(elapsed, 5),
                        "n_runs_eval": N_RUNS_SENSITIVITY,
                        "n_runs_greedy_oracle": N_RUNS_SENSITIVITY,
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

    header = (
        "Severity sensitivity study\n"
        f"propagation_modes={len(PROPAGATION_MODES)} severity_regimes={len(SEVERITY_REGIMES)} n_runs={N_RUNS_SENSITIVITY}\n"
        "Severity affects evaluation in all rows; propagation scaling is optional by mode.\n\n"
    )
    out_log.write_text(header + "\n".join(logs), encoding="utf-8")
    logger.info("Wrote %s (%d rows)", out_csv, len(all_rows))


if __name__ == "__main__":
    main()
