# Severity-Aware Public-Health Misinformation Intervention

This project studies budgeted misinformation containment on social graphs. It starts with a standard binary influence-blocking setup and extends it to a severity-aware public-health harm objective. The implementation is a reproducible semi-synthetic simulation pipeline: CoAID provides COVID-19 misinformation labels, article metadata, and engagement signals; SNAP ego-Twitter provides real graph topology and structural communities; diffusion and interventions are simulated with fixed random seeds.

The project does **not** claim that CoAID articles were posted by the SNAP ego-network users, and it does **not** estimate a real Twitter diffusion process. The defensible question is narrower: given realistic content-derived severity regimes and realistic graph topology, which node-blocking strategies reduce spread, severity-weighted harm, and concentrated harm inside structural network silos?

## Main Contribution

The project extends binary misinformation containment from:

\[
\min_B \mathbb{E}[\text{infected nodes}]
\]

to severity-aware harm minimization:

\[
\min_B \mathbb{E}[\text{severity-weighted total harm}]
\]

and a resilience variant:

\[
\mathbb{E}[\text{total harm}] + \lambda \cdot \mathbb{E}[\text{worst-community harm rate}]
\]

This lets the system ask not only "how many nodes received misinformation?", but also "how harmful was the misinformation?" and "did harm concentrate inside a structural community?"

## Repository Layout

```text
.
├── README.md                         # This consolidated project guide
├── requirements.txt                  # Python environment
├── scripts/reproduce_final_results.sh # One-command final reproduction
├── src/
│   ├── config.py                     # Central paths
│   ├── data/                         # CoAID, Twitter, scenario, severity builders
│   ├── models/                       # Diffusion, interventions, harm, severity
│   └── experiments/                  # Experiment runners and figure builders
├── data/
│   ├── raw/coaid                     # CoAID CSVs, usually symlinked to archive-2/
│   ├── raw/twitter                   # SNAP ego-Twitter files, usually symlinked to twitter/
│   └── processed/                    # Generated processed tables and ego artifacts
└── outputs/
    ├── tables/                       # Generated result CSVs
    ├── figures/                      # Generated report figures
    └── logs/                         # Run logs
```

## Environment Setup

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The required packages are listed in `requirements.txt`:

```text
pandas>=2.0
numpy>=1.24
networkx>=3.0
matplotlib>=3.7
scikit-learn>=1.3
scipy>=1.11
tqdm>=4.66
jupyter>=1.0
```

## Data Setup

Download the two raw datasets from their original sources:

- CoAID COVID-19 misinformation data: [CoAID GitHub repository](https://github.com/cuilimeng/CoAID)
- SNAP ego-Twitter graph data: [SNAP Twitter ego networks](https://snap.stanford.edu/data/ego-Twitter.html)

Expected raw data layout:

```text
data/raw/coaid    -> CoAID CSV files
data/raw/twitter  -> SNAP ego-Twitter files
```

In this workspace, those paths are symlinked to local extracted folders:

```text
data/raw/coaid   -> archive-2/
data/raw/twitter -> twitter/
```

If you start from the original downloads, extract the CoAID archive and SNAP `twitter.tar.gz` / `twitter.tar`, then either place the extracted contents at `data/raw/coaid` and `data/raw/twitter`, or recreate equivalent symlinks:

```bash
mkdir -p data/raw
ln -s ../../archive-2 data/raw/coaid
ln -s ../../twitter data/raw/twitter
```

## One-Command Reproduction

After installing dependencies and confirming the raw data paths, run:

```bash
./scripts/reproduce_final_results.sh
```

This script rebuilds the processed data, runs validation checks, executes the binary and harm-aware experiment suites, and regenerates report tables and figures.

It runs the following stages:

```bash
python -m src.data.build_coaid_tables
python -m src.data.build_twitter_graphs
python -m src.data.build_scenarios
python -m src.data.export_selected_egos
python -m src.models.test_diffusion
python -m src.experiments.run_smoke_baselines
python -m src.experiments.run_intervention_comparison
python -m src.experiments.run_full_matrix
python -m src.experiments.run_fairness_robustness
python -m src.experiments.make_figures
python -m src.experiments.run_final_harm_package
python -m src.experiments.make_report_topology_figures
```

Main outputs:

```text
outputs/tables/full_experiment_results.csv
outputs/tables/fairness_robustness_results.csv
outputs/tables/harm_story_sweep_results.csv
outputs/tables/table_harm_story_primary_slice.csv
outputs/tables/table_harm_story_blocked_set_changes.csv
outputs/tables/table_harm_story_budget_transparency.csv
outputs/figures/
outputs/logs/
```

## Reproducibility and Random Seeds

The experiment scripts use fixed seeds for seed selection, random blocking, greedy oracle simulation, and final evaluation.

Important constants include:

- `MATRIX_EVAL_SEED = 314159265` in `src/experiments/run_full_matrix.py`
- `EVAL_SEED = 927172` in `src/experiments/run_fairness_robustness.py`
- `EVAL_SEED = 818181` in `src/experiments/run_harm_story_sweep.py`
- Deterministic per-cell seeds generated by SHA-256 hashes of ego id, seed strategy, propagation regime, severity regime, and budget
- `PYTHONHASHSEED=0` set by `scripts/reproduce_final_results.sh`
- `networkx.spring_layout(..., seed=42)` for report topology visualizations

Monte Carlo replicate `i` uses `random_seed + i` inside the Independent Cascade simulator, so repeated runs with the same inputs reproduce the same CSV values.

## Data Processing

### CoAID

`src/data/build_coaid_tables.py` builds unified article and engagement tables from CoAID news files.

Key rules:

- Fake and real records are keyed by `(label, article_id)` because numeric article ids can overlap across labels.
- Fake `_5` and `_7` splits are concatenated.
- Real `_5` and `_7` splits are concatenated and deduplicated by `article_id`.
- Engagement is aggregated from tweet and reply tables:
  - `n_related_tweets`
  - `n_unique_tweets`
  - `n_replies`
  - `n_unique_reply_tweets`
  - `engagement_total = n_related_tweets + n_replies`

Generated files include:

```text
data/processed/coaid_articles.csv
data/processed/coaid_engagement.csv
data/processed/coaid_articles_with_engagement.csv
outputs/tables/coaid_summary.csv
outputs/tables/coaid_missingness.csv
```

Current processed summary:

- 838 fake articles
- 2,717 real articles
- 3,555 total articles
- 2,614 article-level engagement rows

### SNAP Ego-Twitter

`src/data/build_twitter_graphs.py` catalogs SNAP ego networks and exports a development graph. `src/data/export_selected_egos.py` exports the selected multi-ego set used in the full matrix.

Key modeling choices:

- Edges are loaded as an undirected simple `networkx.Graph`.
- Self-loops and duplicate edges are ignored.
- SNAP `.circles` files provide structural community labels when non-empty.
- If circles are empty, `src/data/community_detection.py` uses NetworkX greedy modularity communities as a fallback.
- Communities are structural graph groups only, not demographics or protected attributes.

Current catalog summary:

- 973 ego networks
- median edge count around 1,270
- 938 egos with non-empty circle files
- 35 egos with empty circle files

Selected ego labels:

```text
dev
medium_circles
moderate_large_circles
detected_communities
large_circles
```

The final headline harm result uses `large_circles`.

## Scenario Design

Each experiment cell binds:

- an ego graph
- structural community labels
- a seed strategy
- number of misinformation seeds
- propagation regime
- Independent Cascade probability `p`
- requested blocking budget
- intervention method
- optional severity regime and propagation multiplier

Seed strategies:

- `random`: uniformly sample seed nodes.
- `high_degree`: choose highest-degree nodes.
- `community_concentrated`: choose the largest structural community and place seeds on high-degree nodes inside that community.

Important: `community_concentrated` does not learn a concentration label per node. It is a scenario design choice. Nodes get structural community IDs from SNAP circles or detected communities; the largest community is selected; high-degree nodes inside it become misinformation seeds.

Propagation regimes:

```text
low    p = 0.01
medium p = 0.03
high   p = 0.05
```

These probabilities are scenario knobs for stress testing. They are not fitted retweet probabilities.

## Diffusion Model

`src/models/diffusion.py` implements Independent Cascade diffusion.

At each wave:

1. Newly infected nodes attempt to infect each susceptible neighbor once.
2. Each attempt succeeds with probability `p`, optionally scaled by a severity propagation multiplier.
3. Blocked nodes are skipped: they cannot start infected, become infected, or transmit.
4. The process stops when no new nodes activate.

The simulator reports:

- `mean_infected`
- `std_infected`
- `mean_infection_rate`
- `mean_steps`
- `community_mean_infection_rate`
- `mean_worst_community_infection_rate`
- `mean_total_harm`
- `mean_harm_per_node`
- `community_mean_harm_rate`
- `mean_worst_community_harm_rate`

## Intervention Methods

Implemented in `src/models/interventions.py`.

Baseline methods:

- `no_intervention`: block nothing.
- `random_blocking`: uniformly sample blockers among non-seed nodes.
- `top_degree_blocking`: block highest-degree non-seed nodes.
- `pagerank_blocking`: block highest-PageRank non-seed nodes.

Greedy methods:

- `greedy_blocking`: greedily minimizes expected infected count.
- `fairness_aware_greedy_blocking`: minimizes infected count plus a worst-community infection penalty.
- `harm_aware_greedy_blocking`: minimizes expected total harm.
- `harm_aware_resilience_greedy_blocking`: minimizes total harm plus a worst-community harm penalty.

Greedy methods use a restricted candidate pool, the union of top-degree and top-PageRank candidates, excluding seeds. This keeps the Monte Carlo oracle computationally feasible. Output tables include `budget_requested`, `budget_sanitized`, `budget_k`, and `greedy_pool_size` so effective budget shortfalls are visible.

## Severity and Harm

`src/data/build_severity_tables.py` scores CoAID fake-labeled articles with a documented heuristic. The scoring uses:

- high-risk terms, such as vaccine refusal, false cures, treatment claims, bleach/disinfectant-style claims
- medium-risk public-health terms, such as testing, symptoms, infection, quarantine, hospital, spread
- guidance/action language, such as should, must, take, stop, avoid, refuse, prevent
- a small engagement adjustment based on fake-article engagement tertiles

Severity is scenario-level, not node-level. It is a plausible stress-test rubric, not a validated clinical harm taxonomy.

Current severity regimes:

```text
low:    664 articles, mean score 0.2067, harm weight 1.0
medium: 148 articles, mean score 0.4760, harm weight 1.6
high:    26 articles, mean score 0.7188, harm weight 2.4
```

Total harm is computed as:

\[
\text{total harm} = \sum_{v \in \text{infected}} \text{harm weight}(v)
\]

In the current main experiments, the harm weight is a scenario-level scalar. Optional node-specific harm weights are supported by the API for future extensions.

## Experiment Suites

### Smoke Baseline

```bash
python -m src.models.test_diffusion
python -m src.experiments.run_smoke_baselines
```

Validates that diffusion and simple blocking behave as expected on the development ego graph.

### Intervention Validation

```bash
python -m src.experiments.run_intervention_comparison
```

Compares no intervention, random, top-degree, PageRank, and greedy methods on a validation scenario.

### Full Binary Matrix

```bash
python -m src.experiments.run_full_matrix
python -m src.experiments.make_figures
```

Runs five ego settings, three seed strategies, three propagation regimes, three budgets, and six binary intervention methods.

Current full matrix summary:

- 135 experiment cells
- 810 method rows
- no intervention average mean infected: 52.708
- random blocking: 46.704
- top-degree blocking: 36.937
- PageRank blocking: 37.227
- greedy blocking: 38.130
- fairness-aware greedy: 38.161

Top-degree is best in 82 cells, PageRank in 22, greedy in 18, and fairness-aware greedy in 13. This is important: simple topology-aware baselines are genuinely strong.

### Fairness Robustness

```bash
python -m src.experiments.run_fairness_robustness
```

Sweeps `lambda_fair` over structural-community penalties. The goal is to test whether penalizing worst-community infection changes the total-spread versus concentrated-spread tradeoff.

### Severity and Harm Package

```bash
python -m src.experiments.run_final_harm_package
python -m src.experiments.make_report_topology_figures
```

Builds severity tables, harm scenarios, harm method comparisons, severity sensitivity experiments, targeted harm-story sweep results, and final report figures.

## Headline Results

The final primary slice is:

```text
ego_label = large_circles
seed_strategy = community_concentrated
propagation_regime = high
p = 0.05
propagation_mode = constant
severity_regime = high
severity_score = 0.695
base_harm_weight = 2.4
budget_requested = 10
n_runs = 32
```

Primary result table:

| Method | Mean infected | Mean total harm | Worst-community harm rate |
|---|---:|---:|---:|
| harm-aware greedy | 65.56 | 157.35 | 1.171 |
| top-degree | 67.69 | 162.45 | 1.351 |
| harm-aware resilience greedy | 67.84 | 162.83 | 1.195 |
| binary greedy | 68.50 | 164.40 | 1.411 |
| fairness-aware greedy | 69.31 | 166.35 | 1.192 |
| PageRank | 69.53 | 166.88 | 1.201 |
| random | 91.28 | 219.08 | 1.351 |
| no intervention | 97.91 | 234.98 | 1.720 |

Compared with binary greedy in the same scenario, harm-aware greedy:

- reduces mean total harm from 164.40 to 157.35
- reduces worst-community harm rate from 1.411 to 1.171
- chooses a different blocked set, with Jaccard overlap 0.667 against binary greedy

This supports the main claim: severity-aware optimization can change actual intervention decisions and improve harm outcomes in high-severity, community-concentrated settings.

## Report Figures

Generated figures include:

```text
outputs/figures/report_topology_communities_large_circles.png
outputs/figures/report_topology_seeds_harm_blockers.png
outputs/figures/report_topology_blocker_overlap.png
outputs/figures/report_community_size_seed_concentration.png
outputs/figures/report_primary_slice_harm_bars.png
```

Recommended use:

- topology/community figure in methodology
- seed and blocker overlay in intervention explanation
- blocker overlap figure in results
- community-size figure for explaining concentrated seeding
- harm bar chart for the primary result

## Methodological Caveats

Keep these caveats explicit when interpreting results:

- CoAID and SNAP are not user-linked.
- Diffusion is simulated with Independent Cascade, not observed in the wild.
- IC probability `p` is a scenario parameter, not a fitted causal estimate.
- Severity is a documented heuristic, not a clinical ground-truth label.
- Structural communities are network silos, not demographic groups.
- Severity-dependent propagation is a sensitivity analysis, not learned virality.
- Greedy-family methods use a restricted candidate pool, so large requested budgets may produce effective budget shortfalls.
- Small gaps among top-degree, PageRank, and greedy-family methods should be interpreted with Monte Carlo uncertainty in mind.

## References

- Cui, L. and Lee, D. CoAID: COVID-19 Healthcare Misinformation Dataset. 2020.
- Leskovec, J. and McAuley, J. Learning to Discover Social Circles in Ego Networks. NeurIPS, 2012.
- Kempe, D., Kleinberg, J., and Tardos, E. Maximizing the Spread of Influence through a Social Network. KDD, 2003.
- Goldenberg, J., Libai, B., and Muller, E. Talk of the Network: A Complex Systems Look at the Underlying Process of Word-of-Mouth. Marketing Letters, 2001.
- Page, L., Brin, S., Motwani, R., and Winograd, T. The PageRank Citation Ranking: Bringing Order to the Web. 1999.
