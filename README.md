# pd-linucb-delayed-budgets (paper_cbwk_delays)

Prototype repository for a paper on contextual bandits with a budget constraint (BwK/CBwK) and delayed feedback.
The main focus is a reproducible large-scale empirical benchmark on Criteo Attribution with `numpy.memmap`,
a strict stop-at-budget feasible-set protocol, and comparisons between:

- Disjoint LinUCB
- Primal-Dual LinUCB (PD-LinUCB)
- CostNormUCB (ratio/sub)
- Context-free PD-BwK

## Important scientific notes

This repo implements a semi-synthetic simulator:

- Contexts `x_t` are sampled from the logged Criteo dataset, with a default temporal train/test split.
- Rewards can be generated from either:
  - an arm-specific clipped-linear ridge model `clip(theta_a^T x,0,1)`, or
  - an arm-specific logistic model `sigmoid(theta_a^T x + b_a)`.
- The online budget controller uses arm-level mean costs `c(a)` computed from the `cost` field
  and normalized to mean approximately 1. This is not per-event counterfactual cost control.
- Delays:
  - For `r=1`, delays are sampled from empirical positive conversion delays, either globally or arm-conditionally.
  - For `r=0`, the environment uses censoring: delay equals `censor_steps = ceil(W/Δ)`.
  - In the default paper setting, `W/Δ = 5000` at `Δ=3600`, so `W=18,000,000` seconds
    (about 208.3 days).
- The budget-sweep script now uses nested gamma tuning:
  - tune `gamma` on a separate context split / seed set,
  - evaluate the selected `gamma*` on held-out seeds.

## Paper artifacts

- `paper_artifacts/figures/` contains figures for Overleaf.
- `paper_artifacts/tables/` contains LaTeX tabulars such as `main_ci.tex`.
- `results/` can still be used as local scratch output, but the paper-facing scripts now write to `paper_artifacts/` by default.

## Environment setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -r requirements.txt
```

Optional, only if you use the Open Bandit Dataset helper in `src/env/make_obd_feedback.py`:

```bash
python -m pip install -r requirements-obd.txt
```

## 1) Download the official Criteo Attribution dataset

Official source page:

- `https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/`

Direct archive used by the helper:

- `https://go.criteo.net/criteo-research-attribution-dataset.zip`

The public dataset page states the archive is released under `CC BY-NC-SA 4.0`. Review the terms on the official page before downloading.

Automatic download into `data/raw/criteo_attrib/`:

```bash
python -m src.env.download_criteo_attribution_dataset \
  --out_dir data/raw/criteo_attrib \
  --accept_criteo_nc_sa_license
```

This extracts at least:

- `data/raw/criteo_attrib/criteo_attribution_dataset.tsv.gz`
- `data/raw/criteo_attrib/README.md`
- `data/raw/criteo_attrib/Experiments.ipynb`

If the automatic download fails because the upstream URL changes, open the official source page above and download the archive manually into `data/raw/criteo_attrib/`.

## 2) Build a train/test memmap artifact

For a leakage-clean split, fit the simulator on earlier rows and evaluate on later rows. The command below creates one artifact with:

- full row memmaps (`X.npy`, `A.npy`, `R.npy`, `C.npy`, `D.npy`)
- `split.npy` with `0=train`, `1=test`
- train-only fit stats in `meta_and_stats.npz`
- train-only `costs_by_arm.npy`, `delays_pos.npy`, and `delays_pos_by_arm.npz`

```bash
python -m src.env.make_criteo_attrib_memmap_full \
  --inp data/raw/criteo_attrib/criteo_attribution_dataset.tsv.gz \
  --out_dir data/processed/criteo_full_k50_d64_real_split80 \
  --k_cap 50 --d_hash 64 \
  --delta_seconds 3600 \
  --censor_seconds $((5000*3600)) \
  --d_max 5000 \
  --train_frac 0.8 \
  --split_mode temporal \
  --split_seed 123
```

Optional: precompute explicit arm-wise ridge parameters on the train split. This is not required, because `meta_and_stats.npz` already contains train-only sufficient statistics.

```bash
python -m src.env.compute_arm_ridge_stats_from_memmap \
  --memmap_dir data/processed/criteo_full_k50_d64_real_split80 \
  --split train \
  --lam 1.0 --out arm_ridge_stats.npz
```

Optional: fit an arm-wise logistic simulator on the train split. This is the cleaner reward model if you want held-out calibration and informative negative labels in the simulator.

```bash
python -m src.env.compute_arm_logistic_stats_from_memmap \
  --memmap_dir data/processed/criteo_full_k50_d64_real_split80 \
  --split train \
  --lam 1.0 \
  --max_iter 8 \
  --out arm_logistic_stats.npz
```

## 3) Run held-out simulator diagnostics

Compare the fitted simulator against held-out rows before running policy sweeps:

```bash
python -m src.eval.run_simulator_diagnostics \
  --memmap_dir data/processed/criteo_full_k50_d64_real_split80 \
  --eval_split test \
  --reward_models linear_clip,logistic \
  --artifact_dir paper_artifacts
```

If you did not fit `arm_logistic_stats.npz`, use `--reward_models linear_clip` instead.

Outputs:

- `paper_artifacts/tables/simulator_diagnostics.csv`
- `paper_artifacts/tables/simulator_diagnostics.tex`
- `paper_artifacts/figures/simulator_calibration.png`
- `paper_artifacts/figures/simulator_delay_cdf.png`
- `paper_artifacts/figures/simulator_cost_train_vs_eval.png`
- `paper_artifacts/figures/simulator_delay_by_arm.png`

## 4) Run the experiment suite on held-out contexts

All evaluation scripts support `--context_split auto`, which means:

- if `split.npy` exists, sample contexts from the test split
- otherwise, sample from all rows

To avoid Matplotlib cache warnings in headless runs:

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR="$(pwd)/.mplconfig"
mkdir -p "$MPLCONFIGDIR"
```

### Main comparison: original linear policy family

```bash
python -m src.eval.run_compare_baselines \
  --memmap_dir data/processed/criteo_full_k50_d64_real_split80 \
  --context_split auto \
  --reward_model linear_clip \
  --policy_model linear \
  --T 5000 --budget_ratio 0.7 --stop_at_budget \
  --n_seeds 10 --seed0 123 \
  --artifact_dir paper_artifacts \
  --tag T5000_rho0.7_split80
```

Outputs:

- `paper_artifacts/tables/main_ci.tex`
- `paper_artifacts/figures/baselines_cum_reward_full4_arm.png`
- `paper_artifacts/figures/baselines_cum_cost_full4_arm.png`
- `paper_artifacts/figures/baselines_spent_schedule.png`
- `paper_artifacts/figures/avg_cost_per_step.png`
- `paper_artifacts/figures/baselines_cum_reward_mean_ci.png`
- `paper_artifacts/figures/baselines_cum_cost_mean_ci.png`

### Gamma sweep

```bash
python -m src.eval.run_gamma_sweep \
  --memmap_dir data/processed/criteo_full_k50_d64_real_split80 \
  --context_split auto \
  --reward_model linear_clip \
  --policy_model linear \
  --T 5000 --budget_ratio 0.7 --stop_at_budget \
  --n_seeds 10 --seed0 123
```

Outputs:

- `paper_artifacts/figures/gamma_sweep_rho0.7_T5000/gamma_sweep_raw.csv`
- `paper_artifacts/figures/gamma_sweep_rho0.7_T5000/gamma_sweep_summary.csv`
- `paper_artifacts/figures/gamma_sweep_rho0.7_T5000/gamma_sweep_reward.png`
- `paper_artifacts/figures/gamma_sweep_rho0.7_T5000/gamma_sweep_spent.png`

### Delay vs no-delay ablation

```bash
python -m src.eval.run_delay_ablation \
  --memmap_dir data/processed/criteo_full_k50_d64_real_split80 \
  --context_split auto \
  --reward_model linear_clip \
  --policy_model linear \
  --T 5000 --budget_ratio 0.7 --stop_at_budget \
  --n_seeds 10 --seed0 123 \
  --artifact_dir paper_artifacts
```

Output:

- `paper_artifacts/tables/delay_ablation.tex`

### Budget sweep

```bash
python -m src.eval.run_budget_sweep_baselines \
  --memmap_dir data/processed/criteo_full_k50_d64_real_split80 \
  --context_split auto \
  --reward_model linear_clip \
  --policy_model linear \
  --T 5000 --stop_at_budget \
  --budgets 0.40,0.55,0.70,0.85 \
  --gammas 0,0.1,0.3,1,2,3,5,10 \
  --n_seeds 10 --seed0 123 \
  --n_tune_seeds 10 --tune_seed0 10123 \
  --artifact_dir paper_artifacts
```

Outputs:

- `paper_artifacts/tables/budget_sweep_tuning_raw.csv`
- `paper_artifacts/tables/budget_sweep_tuning_summary.csv`
- `paper_artifacts/tables/budget_sweep_eval_raw.csv`
- `paper_artifacts/tables/budget_sweep_summary.csv`
- `paper_artifacts/tables/budget_sweep.tex`
- `paper_artifacts/figures/budget_sweep_reward.png`
- `paper_artifacts/figures/budget_sweep_gamma_star.png`
- `paper_artifacts/figures/budget_sweep_spent.png`

### Stronger logistic setup

If `arm_logistic_stats.npz` exists and you want the cleaner reward model plus contextual learners that update on delayed zeros, run:

```bash
python -m src.eval.run_compare_baselines \
  --memmap_dir data/processed/criteo_full_k50_d64_real_split80 \
  --context_split auto \
  --reward_model logistic \
  --policy_model logistic \
  --T 5000 --budget_ratio 0.7 --stop_at_budget \
  --n_seeds 10 --seed0 123 \
  --artifact_dir paper_artifacts/logistic \
  --tag T5000_rho0.7_logistic
```

The same `--reward_model logistic --policy_model logistic` pair can be used with `run_gamma_sweep.py`, `run_delay_ablation.py`, and `run_budget_sweep_baselines.py`.

## 5) One-command evaluation runner

After preprocessing, you can run the full suite with:

```bash
bash scripts/run_full_paper_experiments.sh \
  data/processed/criteo_full_k50_d64_real_split80 \
  paper_artifacts
```

By default this runs with `REWARD_MODEL=linear_clip` and `POLICY_MODEL=linear`.
To run the logistic setup end-to-end:

```bash
REWARD_MODEL=logistic POLICY_MODEL=logistic \
bash scripts/run_full_paper_experiments.sh \
  data/processed/criteo_full_k50_d64_real_split80 \
  paper_artifacts/logistic
```

When that finishes, the figures to upload to Overleaf are in `paper_artifacts/figures/`, and the LaTeX tabulars are in `paper_artifacts/tables/`.
