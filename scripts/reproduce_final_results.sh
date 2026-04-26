#!/usr/bin/env bash
set -euo pipefail

# Rebuilds the processed data, validation tables, final experiment tables, and
# report figures used for the final project. Run from any directory.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONHASHSEED=0
export MPLBACKEND=Agg

echo "==> Checking Python environment"
python - <<'PY'
import importlib

required = ["pandas", "numpy", "networkx", "matplotlib", "scipy", "sklearn", "tqdm"]
missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
if missing:
    raise SystemExit(
        "Missing packages: "
        + ", ".join(missing)
        + "\nInstall with: python -m pip install -r requirements.txt"
    )
PY

echo "==> Building processed CoAID and Twitter artifacts"
python -m src.data.build_coaid_tables
python -m src.data.build_twitter_graphs
python -m src.data.build_scenarios
python -m src.data.export_selected_egos

echo "==> Running diffusion checks and baseline validation"
python -m src.models.test_diffusion
python -m src.experiments.run_smoke_baselines
python -m src.experiments.run_intervention_comparison

echo "==> Running binary influence-blocking experiments"
python -m src.experiments.run_full_matrix
python -m src.experiments.run_fairness_robustness
python -m src.experiments.make_figures

echo "==> Running severity and harm-aware experiments"
python -m src.experiments.run_final_harm_package
python -m src.experiments.make_report_topology_figures

echo "==> Final results reproduced"
echo "Tables:  outputs/tables/"
echo "Figures: outputs/figures/"
echo "Logs:    outputs/logs/"
